#requires -Version 5.1
<#
.SYNOPSIS
  Banc d'essai PP/TG pour 6x Polaris — session unique, warm-up, repetitions, CSV.

.DESCRIPTION
  Campagne A (Llama 70B) par defaut. Mesure prompt_per_second et predicted_per_second
  via /v1/chat/completions. Variantes alternees pour limiter la derive thermique.
  Voir docs/Plan Optimiser PP-TG-VRAM.md.
#>
param(
    [ValidateSet("llama70", "gptoss")]
    [string]$Campagne = "llama70",
    # Etiquettes libres separees par virgule (ex: ref,fa,mmq,both) — une ligne CSV par label x run
    [string]$Labels = "ref",
    # Chemins llama-server optionnels, alignes sur $Labels (sinon SERVEUR_LLAMA unique)
    [string]$Serveurs = "",
    [string]$Modele = $(if ($env:MODELE_LLAMA) { $env:MODELE_LLAMA } else { $env:LLAMA_MODEL }),
    [string]$ServeurLlama = $(if ($env:SERVEUR_LLAMA) { $env:SERVEUR_LLAMA } else { $env:LLAMA_SERVER }),
    [string]$Peripheriques = "Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4,Vulkan5",
    [string]$TensorSplit = "",
    [string]$CtxList = "4096,8192,16384,25600",
    [int]$GenTokens = 128,
    # Taille cible du prompt de mesure (tokens approx). 0 = ~75% du ctx (lent / long PP).
    [int]$PromptTokens = 2048,
    [int]$WarmupTokens = 32,
    [int]$Runs = 5,
    [int]$Batch = 2048,
    [int]$UBatch = 512,
    # Balayage -b/-ub : "2048:512,1024:256,..." — labels = b{b}ub{ub}, meme binaire
    [string]$BatchVariants = "",
    [int]$Threads = 8,
    [int]$ThreadsBatch = -1,
    [string]$Numa = "",
    [switch]$TurboQuant,
    # Sans -TurboQuant : KV q8_0/q8_0 (plus sûr que f16 sur Polaris 8 Go)
    [string]$CacheTypeK = "q8_0",
    [string]$CacheTypeV = "q8_0",
    [switch]$Alternate,
    [string]$Hote = "127.0.0.1",
    [int]$Port = 8080,
    [int]$TimeoutChargeSec = 900,
    [int]$TimeoutReqSec = 1200,
    [string]$NoteBuild = ""
)

$ErrorActionPreference = "Stop"
$inv = [System.Globalization.CultureInfo]::InvariantCulture
function F([double]$x, [int]$Digits = 3) {
    return $x.ToString("0." + ("0" * $Digits), $inv)
}
$root = Split-Path $PSScriptRoot -Parent
$dirLogs = Join-Path $root "logs"
$dirResults = Join-Path $root "bench_results"
foreach ($d in @($dirLogs, $dirResults)) {
    if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

if (-not $ServeurLlama) {
    $ServeurLlama = Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant\build\bin\llama-server.exe"
}
if (-not (Test-Path -LiteralPath $ServeurLlama)) {
    Write-Error "llama-server introuvable : $ServeurLlama"
}
if (-not $Modele -or -not (Test-Path -LiteralPath $Modele)) {
    Write-Error "Modele introuvable. Definissez `$env:MODELE_LLAMA."
}

$ServeurLlama = (Resolve-Path -LiteralPath $ServeurLlama).Path
$Modele = (Resolve-Path -LiteralPath $Modele).Path
if ($Runs -lt 1) { Write-Error "-Runs doit etre >= 1" }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$csvPath = Join-Path $dirResults "polaris-$Campagne-$stamp.csv"
$metaPath = Join-Path $dirResults "polaris-$Campagne-$stamp.meta.txt"

$batchMap = @{}  # label -> @{ Batch=..; UBatch=.. }
if ($BatchVariants) {
    $labelList = @()
    foreach ($tok in ($BatchVariants -split ",")) {
        $t = $tok.Trim()
        if (-not $t) { continue }
        $parts = $t.Split(":")
        if ($parts.Count -ne 2) { Write-Error "BatchVariants invalide '$t' (attendu b:ub)" }
        $bVal = [int]$parts[0].Trim()
        $uVal = [int]$parts[1].Trim()
        $lab = "b${bVal}ub${uVal}"
        $labelList += $lab
        $batchMap[$lab] = @{ Batch = $bVal; UBatch = $uVal }
    }
    if ($labelList.Count -eq 0) { Write-Error "-BatchVariants vide apres parse" }
} else {
    $labelList = @()
    foreach ($tok in ($Labels -split ",")) {
        $t = $tok.Trim()
        if ($t) { $labelList += $t }
    }
    if ($labelList.Count -eq 0) { Write-Error "-Labels vide" }
    foreach ($l in $labelList) {
        $batchMap[$l] = @{ Batch = $Batch; UBatch = $UBatch }
    }
}

$serverMap = @{}
if ($Serveurs -and -not $BatchVariants) {
    $parts = @()
    foreach ($tok in ($Serveurs -split ",")) {
        $t = $tok.Trim()
        if ($t) { $parts += $t }
    }
    if ($parts.Count -ne $labelList.Count) {
        Write-Error "-Serveurs doit avoir le meme nombre d'entrees que -Labels"
    }
    for ($i = 0; $i -lt $labelList.Count; $i++) {
        $sp = $parts[$i]
        if (-not (Test-Path -LiteralPath $sp)) { Write-Error "Serveur introuvable pour $($labelList[$i]) : $sp" }
        $serverMap[$labelList[$i]] = (Resolve-Path -LiteralPath $sp).Path
    }
} else {
    if (-not $ServeurLlama) {
        $ServeurLlama = Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant\build\bin\llama-server.exe"
    }
    if (-not (Test-Path -LiteralPath $ServeurLlama)) {
        Write-Error "llama-server introuvable : $ServeurLlama"
    }
    $ServeurLlama = (Resolve-Path -LiteralPath $ServeurLlama).Path
    foreach ($l in $labelList) { $serverMap[$l] = $ServeurLlama }
}

$ctxArr = @()
foreach ($tok in ($CtxList -split ",")) {
    $t = $tok.Trim()
    if ($t) { $ctxArr += [int]$t }
}

function Get-ExeMeta([string]$Path) {
    $obj = [ordered]@{
        Path = $Path
        Length = (Get-Item -LiteralPath $Path).Length
        LastWrite = (Get-Item -LiteralPath $Path).LastWriteTime.ToString("s")
        Sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        VersionText = ""
    }
    try {
        $v = & $Path --version 2>&1 | Out-String
        $obj.VersionText = ($v -replace "\s+", " ").Trim()
    } catch { }
    return [pscustomobject]$obj
}

function Stop-LlamaServer {
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

function Wait-ServerReady {
    param([int]$TimeoutSec, [string]$LogPath, [int]$ProcId)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $alive = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
        if (-not $alive) {
            throw "llama-server termine prematurement. Voir $LogPath"
        }
        if (Test-Path -LiteralPath $LogPath) {
            $tail = Get-Content -LiteralPath $LogPath -Tail 120 -ErrorAction SilentlyContinue
            $joined = ($tail -join "`n")
            # OOM transitoire puis retry (ex. sans pipeline parallelism) = OK si le process vit.
            # Ne tuer que les asserts fatals / device lost non recuperes.
            if ($joined -match "GGML_ASSERT") {
                throw "Echec chargement (GGML_ASSERT). Voir $LogPath"
            }
            if ($joined -match "ErrorDeviceLost" -and $joined -notmatch "server is listening|HTTP server is listening") {
                # laisser une chance au health check ; si device lost et pas encore pret, continuer jusqu'a mort process / timeout
            }
        }
        try {
            $h = Invoke-RestMethod -Uri "http://${Hote}:${Port}/health" -TimeoutSec 5
            if ($h) { return }
        } catch { }
        Start-Sleep -Seconds 5
    }
    throw "Timeout ${TimeoutSec}s. Voir $LogPath"
}

function Get-BufferSizes {
    param([string]$LogPath)
    $map = @{}
    if (-not (Test-Path -LiteralPath $LogPath)) { return $map }
    Select-String -LiteralPath $LogPath -Pattern "(\S+) model buffer size =\s+([\d.]+) MiB" | ForEach-Object {
        $map[$_.Matches[0].Groups[1].Value] = [double]$_.Matches[0].Groups[2].Value
    }
    return $map
}

function New-PromptText {
    param([int]$ApproxTokens)
    # ~1 token anglais ~ 4 chars ; pad deterministe pour viser la taille PP
    $unit = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa. "
    $targetChars = [Math]::Max(64, $ApproxTokens * 4)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("Resume succinctement le texte suivant en francais (benchmark PP/TG Polaris).`n`n")
    while ($sb.Length -lt $targetChars) { [void]$sb.Append($unit) }
    return $sb.ToString()
}

function Invoke-ChatBench {
    param(
        [string]$Label,
        [string]$Prompt,
        [int]$NPredict
    )
    # cache_prompt=false : sinon llama-server reutilise le KV et prompt_n tombe a 1 aux runs suivants
    $bodyObj = @{
        messages     = @(@{ role = "user"; content = $Prompt })
        max_tokens   = $NPredict
        temperature  = 0
        stream       = $false
        cache_prompt = $false
    }
    $body = $bodyObj | ConvertTo-Json -Depth 6 -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "http://${Hote}:${Port}/v1/chat/completions" -Method Post `
        -Body $body -ContentType "application/json; charset=utf-8" -TimeoutSec $TimeoutReqSec
    $sw.Stop()
    $t = $r.timings
    if (-not $t) { throw "$Label : pas de timings dans la reponse" }
    return [pscustomobject]@{
        Label       = $Label
        ElapsedSec  = [math]::Round([double]$sw.Elapsed.TotalSeconds, 1)
        PromptTokS  = [math]::Round([double]$t.prompt_per_second, 3)
        GenTokS     = [math]::Round([double]$t.predicted_per_second, 3)
        PromptN     = [int]$t.prompt_n
        PredictedN  = [int]$t.predicted_n
        PromptMs    = [math]::Round([double]$t.prompt_ms, 1)
        PredictedMs = [math]::Round([double]$t.predicted_ms, 1)
    }
}

function Format-BenchLine {
    param($Lab, $Ctx, $Run, $Phase, $Meas, $Bufs, $Sha, $Status)
    $v = {
        param($k)
        if ($null -eq $Bufs[$k]) { return "" }
        return ([double]$Bufs[$k]).ToString("0.##", $inv)
    }
    return ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18}" -f `
        $Lab, $Ctx, $Run, $Phase,
        (F $Meas.PromptTokS), (F $Meas.GenTokS),
        $Meas.PromptN, $Meas.PredictedN,
        (F $Meas.PromptMs 1), (F $Meas.PredictedMs 1), (F $Meas.ElapsedSec 1),
        (& $v "Vulkan0"), (& $v "Vulkan1"), (& $v "Vulkan2"),
        (& $v "Vulkan3"), (& $v "Vulkan4"), (& $v "Vulkan5"),
        $Sha, $Status)
}

function Start-BenchServer {
    param(
        [string]$Exe,
        [int]$Ctx,
        [int]$BatchSize,
        [int]$UBatchSize,
        [string]$LogPath,
        [string]$OutPath
    )
    $argsList = @(
        "-m", $Modele
        "--host", $Hote, "--port", "$Port"
        "-c", "$Ctx"
        "-ngl", "99"
        "-fa", "on"
        "-np", "1"
        "--split-mode", "layer"
        "--device", $Peripheriques
        "--fit", "off"
        "-b", "$BatchSize"
        "-ub", "$UBatchSize"
        "-t", "$Threads"
        "-v", "--log-timestamps"
    )
    if ($ThreadsBatch -gt 0) { $argsList += @("-tb", "$ThreadsBatch") }
    if ($TensorSplit) { $argsList += @("--tensor-split", $TensorSplit) }
    if ($Numa) { $argsList += @("--numa", $Numa) }
    if ($TurboQuant) {
        $argsList += @("--cache-type-k", "q8_0", "--cache-type-v", "turbo3")
    } elseif ($CacheTypeK -or $CacheTypeV) {
        if ($CacheTypeK) { $argsList += @("--cache-type-k", $CacheTypeK) }
        if ($CacheTypeV) { $argsList += @("--cache-type-v", $CacheTypeV) }
    }
    # Warm-up serveur desactive : le banc fait son propre warm-up mesure
    $argsList += @("--no-warmup")
    return Start-Process -FilePath $Exe -ArgumentList $argsList `
        -RedirectStandardError $LogPath -RedirectStandardOutput $OutPath `
        -PassThru -WindowStyle Hidden
}

# Schedule: by default label-major; -Alternate = round-robin labels for each (ctx, run)
$schedule = New-Object System.Collections.Generic.List[object]
if ($Alternate -and $labelList.Count -gt 1) {
    for ($r = 1; $r -le $Runs; $r++) {
        foreach ($ctx in $ctxArr) {
            foreach ($lab in $labelList) {
                $schedule.Add([pscustomobject]@{ Label = $lab; Ctx = $ctx; Run = $r; Phase = "meas" })
            }
        }
    }
} else {
    foreach ($lab in $labelList) {
        foreach ($ctx in $ctxArr) {
            for ($r = 1; $r -le $Runs; $r++) {
                $schedule.Add([pscustomobject]@{ Label = $lab; Ctx = $ctx; Run = $r; Phase = "meas" })
            }
        }
    }
}

# Meta
$metas = @()
foreach ($lab in $labelList) {
    $metas += Get-ExeMeta $serverMap[$lab]
}
@(
    "stamp=$stamp"
    "campagne=$Campagne"
    "modele=$Modele"
    "labels=$($labelList -join ',')"
    "ctx=$($ctxArr -join ',')"
    "runs=$Runs"
    "gen_tokens=$GenTokens"
    "prompt_tokens=$PromptTokens"
    "batch=$Batch"
    "ubatch=$UBatch"
    "batch_variants=$BatchVariants"
    "threads=$Threads"
    "threads_batch=$ThreadsBatch"
    "tensor_split=$TensorSplit"
    "devices=$Peripheriques"
    "numa=$Numa"
    "turboquant=$TurboQuant"
    "alternate=$Alternate"
    "note=$NoteBuild"
    "host=$env:COMPUTERNAME"
    "os=$([Environment]::OSVersion.VersionString)"
) + ($metas | ForEach-Object { "exe=$($_.Path); sha256=$($_.Sha256); ver=$($_.VersionText); mtime=$($_.LastWrite)" }) |
    Set-Content -LiteralPath $metaPath -Encoding UTF8

"label,ctx,run,phase,prompt_tok_s,gen_tok_s,prompt_n,predicted_n,prompt_ms,predicted_ms,elapsed_s,vulkan0_mib,vulkan1_mib,vulkan2_mib,vulkan3_mib,vulkan4_mib,vulkan5_mib,server_sha256,status" |
    Set-Content -LiteralPath $csvPath -Encoding UTF8

Write-Host "=== benchmark_polaris ($Campagne) ===" -ForegroundColor Cyan
Write-Host "CSV  : $csvPath"
Write-Host "Meta : $metaPath"
Write-Host "Labels=$($labelList -join ',') Ctx=$($ctxArr -join ',') Runs=$Runs Alternate=$Alternate"
if ($BatchVariants) { Write-Host "BatchVariants=$BatchVariants" }
Write-Host ""

$lastKey = ""
$bufs = @{}
$exeSha = ""

try {
    foreach ($step in $schedule) {
        $lab = $step.Label
        $ctx = [int]$step.Ctx
        $exe = $serverMap[$lab]
        $bSz = [int]$batchMap[$lab].Batch
        $uSz = [int]$batchMap[$lab].UBatch
        $key = "$lab|$ctx|$exe|$bSz|$uSz"

        if ($key -ne $lastKey) {
            Stop-LlamaServer
            $tag = "$lab-c$ctx"
            $logPath = Join-Path $dirLogs "polaris-$tag-$stamp.log"
            $outPath = Join-Path $dirLogs "polaris-$tag-$stamp.out.log"
            Write-Host "---- charge $tag (-b $bSz -ub $uSz) ----" -ForegroundColor Yellow
            $proc = Start-BenchServer -Exe $exe -Ctx $ctx -BatchSize $bSz -UBatchSize $uSz -LogPath $logPath -OutPath $outPath
            Wait-ServerReady -TimeoutSec $TimeoutChargeSec -LogPath $logPath -ProcId $proc.Id
            $bufs = Get-BufferSizes -LogPath $logPath
            $exeSha = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
            Write-Host ("Buffers V0..V5: {0:N0} {1:N0} {2:N0} {3:N0} {4:N0} {5:N0}" -f `
                $bufs["Vulkan0"], $bufs["Vulkan1"], $bufs["Vulkan2"], $bufs["Vulkan3"], $bufs["Vulkan4"], $bufs["Vulkan5"])

            # Warm-up une fois par chargement
            $promptWarm = New-PromptText -ApproxTokens ([Math]::Min(512, [Math]::Max(64, [int]($ctx / 8))))
            Write-Host "Warmup..."
            $warm = Invoke-ChatBench -Label "$tag-warm" -Prompt $promptWarm -NPredict $WarmupTokens
            Write-Host ("  warm PP={0} TG={1}" -f $warm.PromptTokS, $warm.GenTokS)
            Add-Content -LiteralPath $csvPath -Value (Format-BenchLine $lab $ctx 0 "warm" $warm $bufs $exeSha "ok") -Encoding UTF8
            $lastKey = $key
        }

        if ($PromptTokens -gt 0) {
            $approxPrompt = [Math]::Min($PromptTokens, $ctx - $GenTokens - 64)
        } else {
            $approxPrompt = [Math]::Min([int]($ctx * 0.75), $ctx - $GenTokens - 64)
        }
        if ($approxPrompt -lt 128) { $approxPrompt = 128 }
        # Sel unique par run pour eviter tout cache residuel
        $prompt = (New-PromptText -ApproxTokens $approxPrompt) + "`n[bench-nonce $lab c$ctx r$($step.Run) $stamp]"
        Write-Host ("Mesure {0} ctx={1} run={2}/{3} (prompt~{4})..." -f $lab, $ctx, $step.Run, $Runs, $approxPrompt)
        try {
            $meas = Invoke-ChatBench -Label "$lab-c$ctx-r$($step.Run)" -Prompt $prompt -NPredict $GenTokens
            if ($meas.PromptN -lt [Math]::Max(64, [int]($approxPrompt * 0.25))) {
                throw ("prompt_n trop bas ($($meas.PromptN)) — cache KV probable")
            }
            Write-Host ("  PP={0} TG={1} (prompt_n={2})" -f $meas.PromptTokS, $meas.GenTokS, $meas.PromptN) -ForegroundColor Green
            Add-Content -LiteralPath $csvPath -Value (Format-BenchLine $lab $ctx $step.Run "meas" $meas $bufs $exeSha "ok") -Encoding UTF8
        }
        catch {
            $msg = $_.Exception.Message.Replace(",", ";")
            Write-Host "FAIL: $msg" -ForegroundColor Red
            $empty = [pscustomobject]@{ PromptTokS=0; GenTokS=0; PromptN=0; PredictedN=0; PromptMs=0; PredictedMs=0; ElapsedSec=0 }
            Add-Content -LiteralPath $csvPath -Value (Format-BenchLine $lab $ctx $step.Run "meas" $empty $bufs $exeSha "FAIL:$msg") -Encoding UTF8
            $lastKey = ""  # force reload next
        }
    }
}
finally {
    Stop-LlamaServer
}

Write-Host ""
Write-Host "Termine." -ForegroundColor Cyan
Write-Host "CSV  : $csvPath"
Write-Host "Meta : $metaPath"
