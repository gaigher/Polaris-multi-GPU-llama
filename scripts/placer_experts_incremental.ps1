#requires -Version 5.1
<#
.SYNOPSIS
  Placement incremental des couches d'experts MoE (llama.cpp) sur multi-GPU.

.DESCRIPTION
  EXPERIMENTAL / BEST-EFFORT — pas une garantie de perf ni d'absence d'OOM.

  Algoritme :
    1) Detecte les peripheriques (Vulkan*/CUDA*) et la carte couche->GPU
       (logs "layer N assigned to device ...").
    2) Place +1 couche d'experts residente par GPU et par tour (reste en CPU via -ot).
    3) S'arrete par GPU au seuil VRAM ou a l'OOM ; les autres GPU continuent.
    4) Option -ValiderCtx : recharge au contexte cible et recule d'une couche
       sur le GPU fautif si OOM compute.

  Limitations :
    - Windows + llama-server (stderr parse).
    - Pattern -ot par defaut adapte aux tenseurs type GPT-OSS / llama.cpp MoE
      (ffn_*_exps). Autres archis : -PatternOt.
    - Reload complet a chaque tour (pas de deplacement a chaud).
    - Les experts d'une couche sont atomiques (pas d'expert #N isole).

.NOTES
  Variables : $env:SERVEUR_LLAMA (ou LLAMA_SERVER), $env:MODELE_LLAMA (ou LLAMA_MODEL).
  Exemple :
    $env:SERVEUR_LLAMA = "...\llama-server.exe"
    $env:MODELE_LLAMA  = "...\model-00001-of-00002.gguf"
    .\scripts\placer_experts_incremental.ps1 -ValiderCtx 8192 -MoeCache
#>

param(
    [string]$ServeurLlama = $(if ($env:SERVEUR_LLAMA) { $env:SERVEUR_LLAMA } else { $env:LLAMA_SERVER }),
    [string]$Modele = $(if ($env:MODELE_LLAMA) { $env:MODELE_LLAMA } else { $env:LLAMA_MODEL }),
    # Liste explicite, sinon auto via --list-devices (tous Vulkan* / CUDA*)
    [string]$Peripheriques = "",
    [string]$Hote = "127.0.0.1",
    [int]$Port = 8080,
    # Contexte des probes (leger) ; -ValiderCtx pour le contexte reel
    [int]$TailleCtxProbe = 2048,
    [int]$ValiderCtx = 0,
    # Stop quand buffer modele >= SeuilPct% de la VRAM totale du device
    [int]$SeuilPct = 82,
    # Marge supplementaire (MiB) retiree du seuil, surtout pour le 1er GPU (compute)
    [int]$MargeReserveMiB = 400,
    [int]$MargeReservePremierMiB = 1500,
    # Fragment regex apres blk.<id>. — defaut GPT-OSS / MoE llama.cpp
    [string]$PatternOt = "ffn_.*_exps.",
    [string]$BufferCible = "CPU",
    [int]$TimeoutChargeSec = 900,
    [switch]$MoeCache,
    [switch]$Json,
    [string]$FichierResultat = ""
)

$ErrorActionPreference = "Stop"

Write-Host @"
=== placer_experts_incremental.ps1 ===
EXPERIMENTAL — outil d'aide au -ot MoE multi-GPU (llama.cpp).
Resultats dependants du modele, backend, pilotes et VRAM libre.
"@ -ForegroundColor Yellow

if (-not $ServeurLlama) {
    $ServeurLlama = Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant\build\bin\llama-server.exe"
}
if (-not (Test-Path -LiteralPath $ServeurLlama)) {
    Write-Error "llama-server introuvable : $ServeurLlama`nDefinissez `$env:SERVEUR_LLAMA (voir README)."
}
if (-not $Modele -or -not (Test-Path -LiteralPath $Modele)) {
    Write-Error "Modele introuvable. Definissez `$env:MODELE_LLAMA vers le GGUF (shard 00001 si split)."
}

$ServeurLlama = (Resolve-Path -LiteralPath $ServeurLlama).Path
$Modele = (Resolve-Path -LiteralPath $Modele).Path

$dirLogs = Join-Path (Split-Path $PSScriptRoot -Parent) "logs"
if (-not (Test-Path -LiteralPath $dirLogs)) {
    New-Item -ItemType Directory -Path $dirLogs | Out-Null
}
if (-not $FichierResultat) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $FichierResultat = Join-Path $dirLogs "placement-experts-$stamp.txt"
}

function Write-Etat {
    param([string]$Msg)
    Write-Host $Msg
    Add-Content -LiteralPath $FichierResultat -Value $Msg
}

function Stop-LlamaServer {
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Get-LlamaDevices {
    param([string]$Exe)
    $out = & $Exe --list-devices 2>&1 | Out-String
    $devs = [ordered]@{}
    foreach ($line in ($out -split "`r?`n")) {
        # Vulkan0: Name (8192 MiB, 7367 MiB free)
        if ($line -match "^\s*((?:Vulkan|CUDA|Metal)\d+)\s*:\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB free)?") {
            $id = $Matches[1]
            $devs[$id] = [pscustomobject]@{
                Id     = $id
                Nom    = $Matches[2].Trim()
                Total  = [int]$Matches[3]
                Libre  = if ($Matches[4]) { [int]$Matches[4] } else { [int]$Matches[3] }
            }
        }
    }
    if ($devs.Count -eq 0) {
        Write-Error "Aucun peripherique GPU detecte via --list-devices.`n$out"
    }
    return $devs
}

function Get-OtString {
    param($CpuSet, [string]$Pat, [string]$Buf)
    $sorted = @($CpuSet | Sort-Object)
    if ($sorted.Count -eq 0) { return $null }
    $nums = $sorted -join "|"
    return "blk\.($nums)\.$Pat=$Buf"
}

function Get-OtTousExpertsCpu {
    param([string]$Pat, [string]$Buf)
    return ".$Pat=$Buf"
}

function Get-SeuilGpu {
    param($DevInfo, [int]$Index, [int]$Pct, [int]$Marge, [int]$MargePremier)
    $base = [math]::Floor($DevInfo.Total * $Pct / 100.0)
    $m = if ($Index -eq 0) { $MargePremier } else { $Marge }
    return [Math]::Max(512, $base - $m)
}

function Invoke-ProbeCharge {
    param(
        [string]$Ot,
        [string]$LogPath,
        [int]$Ctx,
        [string]$Devices,
        [bool]$AvecMoeCache
    )

    Stop-LlamaServer
    if (Test-Path -LiteralPath $LogPath) { Remove-Item -LiteralPath $LogPath -Force }
    $outPath = "$LogPath.out"
    if (Test-Path -LiteralPath $outPath) { Remove-Item -LiteralPath $outPath -Force }

    $argsList = @(
        "-m", $Modele,
        "--host", $Hote, "--port", "$Port",
        "-c", "$Ctx", "-ngl", "99", "-fa", "on",
        "--split-mode", "layer",
        "--device", $Devices,
        "-np", "1",
        "--fit", "off", "--no-warmup",
        "-v", "--log-timestamps"
    )
    if ($Ot) { $argsList += @("-ot", $Ot) }
    if ($AvecMoeCache) { $argsList += @("--moe-cache", "on") }

    Write-Host ("  Probe ctx={0} ot={1}" -f $Ctx, $(if ($Ot) { $Ot } else { "(aucun)" })) -ForegroundColor DarkCyan
    $p = Start-Process -FilePath $ServeurLlama -ArgumentList $argsList `
        -NoNewWindow -PassThru `
        -RedirectStandardError $LogPath `
        -RedirectStandardOutput $outPath

    $deadline = (Get-Date).AddSeconds($TimeoutChargeSec)
    $etat = "timeout"
    $txt = ""
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        if (Test-Path -LiteralPath $LogPath) {
            $txt = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
        }
        if ($txt -match "model buffer size" -and $txt -match "offloaded \d+/\d+ layers") {
            if ($txt -notmatch "ErrorOutOfDeviceMemory|failed to allocate") {
                $etat = "ok"
                Start-Sleep -Seconds 6
                if (Test-Path -LiteralPath $LogPath) {
                    $txt = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
                }
                break
            }
        }
        if ($txt -match "listening on|model loaded") { $etat = "ok"; break }
        if ($txt -match "ErrorOutOfDeviceMemory|failed to allocate (?:Vulkan|CUDA|Metal)|failed to load model|failed to allocate compute") {
            $etat = "oom"
            break
        }
        try {
            $null = Invoke-RestMethod -Uri "http://${Hote}:$Port/health" -TimeoutSec 1 -ErrorAction Stop
            $etat = "ok"
            break
        } catch { }

        if ($p.HasExited) {
            if ($txt -match "ErrorOutOfDeviceMemory|failed to allocate") { $etat = "oom" }
            elseif ($txt -match "model buffer size") { $etat = "ok" }
            else { $etat = "exit" }
            break
        }
    }

    $buffers = @{}
    $oomGpu = $null
    $layerMap = [ordered]@{}
    if (Test-Path -LiteralPath $LogPath) {
        if (-not $txt) { $txt = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue }
        $bufMatches = Select-String -Path $LogPath -Pattern "((?:Vulkan|CUDA|Metal)\d+)\s+model buffer size =\s+([\d.]+)\s+MiB"
        foreach ($m in $bufMatches) {
            $buffers[$m.Matches[0].Groups[1].Value] = [double]$m.Matches[0].Groups[2].Value
        }
        if ($txt -match "failed to allocate ((?:Vulkan|CUDA|Metal)\d+)") {
            $oomGpu = $Matches[1]
        }
        $layMatches = Select-String -Path $LogPath -Pattern "layer\s+(\d+)\s+assigned to device\s+((?:Vulkan|CUDA|Metal)\d+)"
        foreach ($m in $layMatches) {
            $layerMap[[int]$m.Matches[0].Groups[1].Value] = $m.Matches[0].Groups[2].Value
        }
    }

    Stop-LlamaServer
    return [pscustomobject]@{
        Etat     = $etat
        Buffers  = $buffers
        OomGpu   = $oomGpu
        LayerMap = $layerMap
        Log      = $LogPath
    }
}

# --- Detection peripheriques ---
$tousDevices = Get-LlamaDevices -Exe $ServeurLlama
if (-not $Peripheriques) {
    $Peripheriques = ($tousDevices.Keys -join ",")
}
$listeGpu = @($Peripheriques -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($g in $listeGpu) {
    if (-not $tousDevices.Contains($g)) {
        Write-Error "Peripherique inconnu : $g. Detectes : $($tousDevices.Keys -join ', ')"
    }
}

Write-Etat "=== Placement incremental experts MoE (EXPERIMENTAL) ==="
Write-Etat "Serveur : $ServeurLlama"
Write-Etat "Modele  : $Modele"
Write-Etat "Devices : $Peripheriques"
Write-Etat ("Seuil   : {0}% VRAM - marge {1} MiB (premier GPU -{2} MiB) | probe ctx={3}" -f $SeuilPct, $MargeReserveMiB, $MargeReservePremierMiB, $TailleCtxProbe)
Write-Etat "Pattern : blk.<id>.$PatternOt=$BufferCible"
Write-Etat ""

$seuils = @{}
for ($i = 0; $i -lt $listeGpu.Count; $i++) {
    $g = $listeGpu[$i]
    $seuils[$g] = Get-SeuilGpu -DevInfo $tousDevices[$g] -Index $i -Pct $SeuilPct -Marge $MargeReserveMiB -MargePremier $MargeReservePremierMiB
    Write-Etat ("  {0} : {1} ({2} MiB total, seuil modele {3} MiB)" -f $g, $tousDevices[$g].Nom, $tousDevices[$g].Total, $seuils[$g])
}

# --- Carte couche -> GPU (load tous experts CPU) ---
Write-Etat ""
Write-Etat "--- Detection carte couches / GPU (experts 100% CPU) ---"
$otAllCpu = Get-OtTousExpertsCpu -Pat $PatternOt -Buf $BufferCible
$log0 = Join-Path $dirLogs "probe-detect-layers.log"
$detect = Invoke-ProbeCharge -Ot $otAllCpu -LogPath $log0 -Ctx $TailleCtxProbe -Devices $Peripheriques -AvecMoeCache:$false
if ($detect.Etat -ne "ok" -or $detect.LayerMap.Count -eq 0) {
    Write-Error "Impossible de detecter la repartition des couches (etat=$($detect.Etat)). Voir $log0"
}

# Ignorer la couche output seule si elle n'a pas d'experts typiques : on prend les ids vus
$CouchesParGpu = [ordered]@{}
foreach ($g in $listeGpu) { $CouchesParGpu[$g] = New-Object System.Collections.Generic.List[int] }

$maxBlk = -1
foreach ($kv in $detect.LayerMap.GetEnumerator()) {
    $layer = [int]$kv.Key
    $gpu = [string]$kv.Value
    if (-not $CouchesParGpu.Contains($gpu)) { continue }
    # Souvent layer == n_layer est l'output sans experts ; on la garde quand meme si assignee
    $CouchesParGpu[$gpu].Add($layer)
    if ($layer -gt $maxBlk) { $maxBlk = $layer }
}

# Pour MoE, les experts sont sur blk.0 .. blk.(n_layer-1) ; exclure la derniere si c'est output-only
# Heuristique : si une couche n'apparait que sur le dernier GPU et = max, et que le GPU a deja d'autres couches, on peut la laisser (placement experts no-op si absents)
$ToutesCouches = @()
foreach ($g in $listeGpu) {
    $arr = @($CouchesParGpu[$g] | Sort-Object)
    $CouchesParGpu[$g] = $arr
    Write-Etat ("  {0} couches : {1}" -f $g, ($arr -join ", "))
    $ToutesCouches += $arr
}
$ToutesCouches = @($ToutesCouches | Sort-Object -Unique)

$ExpertsCpu = New-Object "System.Collections.Generic.HashSet[int]"
foreach ($c in $ToutesCouches) { [void]$ExpertsCpu.Add($c) }

$GpuPlein = @{}
$IndexProchaine = @{}
foreach ($gpu in $listeGpu) {
    $GpuPlein[$gpu] = $false
    $IndexProchaine[$gpu] = 0
}

$tour = 0
$dernierOtOk = Get-OtTousExpertsCpu -Pat $PatternOt -Buf $BufferCible
$dernierBuffersOk = @{}

while ($true) {
    $tour++
    $ajouts = @()
    foreach ($gpu in $listeGpu) {
        if ($GpuPlein[$gpu]) { continue }
        $liste = @($CouchesParGpu[$gpu])
        $idx = $IndexProchaine[$gpu]
        if ($idx -ge $liste.Count) {
            $GpuPlein[$gpu] = $true
            Write-Etat "$gpu : plus de couche a placer -> plein"
            continue
        }
        $couche = $liste[$idx]
        [void]$ExpertsCpu.Remove($couche)
        $IndexProchaine[$gpu] = $idx + 1
        $ajouts += "${gpu}:blk.$couche"
    }

    if ($ajouts.Count -eq 0) {
        Write-Etat "Arret : aucun GPU n'accepte plus de couche."
        break
    }

    $ot = Get-OtString -CpuSet $ExpertsCpu -Pat $PatternOt -Buf $BufferCible
    $log = Join-Path $dirLogs ("probe-tour-{0:00}.log" -f $tour)
    $residentsNow = @($ToutesCouches | Where-Object { -not $ExpertsCpu.Contains($_) } | Sort-Object)
    Write-Etat ""
    Write-Etat ("----- Tour {0} : + {1} -----" -f $tour, ($ajouts -join ", "))
    Write-Etat ("Experts GPU (residents) : {0}" -f ($residentsNow -join ", "))
    Write-Etat "-ot : $ot"

    $res = Invoke-ProbeCharge -Ot $ot -LogPath $log -Ctx $TailleCtxProbe -Devices $Peripheriques -AvecMoeCache:$false

    if ($res.Etat -ne "ok") {
        Write-Etat ("ECHEC ({0}) - annulation des ajouts de ce tour." -f $res.Etat)
        if ($res.OomGpu) { Write-Etat ("  OOM sur {0}" -f $res.OomGpu) }
        foreach ($a in $ajouts) {
            if ($a -match "^((?:Vulkan|CUDA|Metal)\d+):blk\.(\d+)$") {
                $g = $Matches[1]
                $c = [int]$Matches[2]
                [void]$ExpertsCpu.Add($c)
                $IndexProchaine[$g] = [Math]::Max(0, $IndexProchaine[$g] - 1)
                if ($res.Etat -eq "oom" -and $res.OomGpu -and ($g -eq $res.OomGpu)) {
                    $GpuPlein[$g] = $true
                    Write-Etat ("  {0} marque plein (couche {1} reste CPU)" -f $g, $c)
                } elseif ($res.Etat -eq "oom" -and -not $res.OomGpu) {
                    $GpuPlein[$g] = $true
                    Write-Etat ("  {0} marque plein par precaution (couche {1} reste CPU)" -f $g, $c)
                } else {
                    Write-Etat ("  {0} : couche {1} retiree pour retry" -f $g, $c)
                }
            }
        }
        if ($res.Etat -ne "oom") {
            Write-Etat "Arret : echec non-OOM (timeout/exit)."
            break
        }
        $encore = $false
        foreach ($gpu in $listeGpu) {
            if ((-not $GpuPlein[$gpu]) -and ($IndexProchaine[$gpu] -lt @($CouchesParGpu[$gpu]).Count)) {
                $encore = $true
            }
        }
        if (-not $encore) { break }
        continue
    }

    $dernierOtOk = $ot
    $dernierBuffersOk = $res.Buffers
    foreach ($gpu in $listeGpu) {
        $mib = 0.0
        if ($res.Buffers.ContainsKey($gpu)) { $mib = $res.Buffers[$gpu] }
        $seuil = $seuils[$gpu]
        $flag = ""
        if ($mib -ge $seuil) {
            $GpuPlein[$gpu] = $true
            $flag = " PLEIN(seuil)"
        }
        Write-Etat ("  {0} model buffer = {1} MiB / seuil {2}{3}" -f $gpu, ([math]::Round($mib, 2)), $seuil, $flag)
    }

    $tousPleins = $true
    foreach ($gpu in $listeGpu) {
        if (-not $GpuPlein[$gpu]) { $tousPleins = $false; break }
    }
    if ($tousPleins) {
        Write-Etat "Tous les GPU sont au seuil ou sans couche restante."
        break
    }
}

# --- Validation contexte cible (recul asymetrique si OOM) ---
if ($ValiderCtx -gt 0 -and $dernierOtOk) {
    Write-Etat ""
    Write-Etat "--- Validation ctx=$ValiderCtx ---"
    $maxEssais = $ToutesCouches.Count
    for ($essai = 0; $essai -lt $maxEssais; $essai++) {
        $logV = Join-Path $dirLogs ("probe-validate-ctx-{0}-try{1}.log" -f $ValiderCtx, $essai)
        $resV = Invoke-ProbeCharge -Ot $dernierOtOk -LogPath $logV -Ctx $ValiderCtx -Devices $Peripheriques -AvecMoeCache:$MoeCache
        if ($resV.Etat -eq "ok") {
            Write-Etat "Validation OK a ctx=$ValiderCtx"
            $dernierBuffersOk = $resV.Buffers
            foreach ($gpu in $listeGpu) {
                $mib = 0.0
                if ($resV.Buffers.ContainsKey($gpu)) { $mib = $resV.Buffers[$gpu] }
                Write-Etat ("  {0} model buffer = {1} MiB" -f $gpu, ([math]::Round($mib, 2)))
            }
            break
        }
        Write-Etat ("Validation ECHEC ({0}){1}" -f $resV.Etat, $(if ($resV.OomGpu) { " sur $($resV.OomGpu)" } else { "" }))
        $cible = $resV.OomGpu
        if (-not $cible) { $cible = $listeGpu[0] }
        # Reculer : remettre en CPU la derniere couche residente de ce GPU
        $residentsGpu = @($CouchesParGpu[$cible] | Where-Object { -not $ExpertsCpu.Contains($_) } | Sort-Object)
        if ($residentsGpu.Count -eq 0) {
            Write-Etat "Impossible de reculer davantage sur $cible"
            break
        }
        $aRetirer = $residentsGpu[-1]
        [void]$ExpertsCpu.Add($aRetirer)
        $dernierOtOk = Get-OtString -CpuSet $ExpertsCpu -Pat $PatternOt -Buf $BufferCible
        Write-Etat ("Recul : blk.{0} -> CPU sur {1}" -f $aRetirer, $cible)
        Write-Etat "-ot : $dernierOtOk"
    }
}

$residents = @($ToutesCouches | Where-Object { -not $ExpertsCpu.Contains($_) } | Sort-Object)
$moeFlag = if ($MoeCache) { " --moe-cache on" } else { "" }
$ctxLaunch = if ($ValiderCtx -gt 0) { $ValiderCtx } else { $TailleCtxProbe }

Write-Etat ""
Write-Etat "=== Resultat final (EXPERIMENTAL) ==="
Write-Etat ("Couches experts RESIDENTES GPU : {0}" -f ($residents -join ", "))
Write-Etat ("Couches experts CPU            : {0}" -f (@($ExpertsCpu | Sort-Object) -join ", "))
Write-Etat "Commande -ot :"
Write-Etat $dernierOtOk
Write-Etat ""
Write-Etat "Exemple de lancement :"
Write-Etat "& `$env:SERVEUR_LLAMA -m `$env:MODELE_LLAMA --host $Hote --port $Port -c $ctxLaunch -ngl 99 -fa on -np 1"
Write-Etat "  --split-mode layer --device $Peripheriques"
Write-Etat ("  -ot `"{0}`"{1} --fit off -v --log-timestamps" -f $dernierOtOk, $moeFlag)
Write-Etat ""
Write-Etat "Detail : $FichierResultat"

if ($Json) {
    $obj = [ordered]@{
        experimental = $true
        ot           = $dernierOtOk
        residents    = @($residents)
        cpu_experts  = @($ExpertsCpu | Sort-Object)
        devices      = $listeGpu
        buffers_mib  = $dernierBuffersOk
        ctx_probe    = $TailleCtxProbe
        ctx_valid    = $(if ($ValiderCtx -gt 0) { $ValiderCtx } else { $null })
        pattern      = $PatternOt
        result_file  = $FichierResultat
    }
    $jsonPath = [System.IO.Path]::ChangeExtension($FichierResultat, ".json")
    ($obj | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    Write-Etat "JSON : $jsonPath"
}

Write-Host ""
Write-Host "Termine. Voir $FichierResultat" -ForegroundColor Green
