# Lance llama-server : 6× Polaris, Llama 70B Q4, split layer, ctx 25.6k
# Par défaut : sans TurboQuant. -TurboQuant : K q8_0 / V turbo3 (nécessite patch Polaris).
param(
    [string]$ServeurLlama = $(if ($env:SERVEUR_LLAMA) { $env:SERVEUR_LLAMA } else { $env:LLAMA_SERVER }),
    [string]$Modele = $(if ($env:MODELE_LLAMA) { $env:MODELE_LLAMA } else { $env:LLAMA_MODEL }),
    [string]$Peripheriques = "Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4,Vulkan5",
    [string]$Hote = "127.0.0.1",
    [int]$Port = 8080,
    [int]$TailleCtx = 25600,
    [switch]$TurboQuant
)

$ErrorActionPreference = "Stop"

if (-not $ServeurLlama) {
    $ServeurLlama = Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant\build\bin\llama-server.exe"
}
if (-not (Test-Path -LiteralPath $ServeurLlama)) {
    Write-Error "llama-server introuvable : $ServeurLlama`nDéfinissez `$env:SERVEUR_LLAMA vers un binaire Vulkan (voir README, section Obtenir llama-server)."
}
if (-not $Modele -or -not (Test-Path -LiteralPath $Modele)) {
    Write-Error "Modèle introuvable. Définissez `$env:MODELE_LLAMA vers un Llama 3.3 70B Q4 local (.gguf)."
}

$ServeurLlama = (Resolve-Path -LiteralPath $ServeurLlama).Path
$Modele = (Resolve-Path -LiteralPath $Modele).Path

Write-Host "Périphériques Vulkan disponibles :" -ForegroundColor Cyan
& $ServeurLlama --list-devices
Write-Host ""

$arguments = @(
    "-m", $Modele,
    "--host", $Hote, "--port", "$Port",
    "-c", "$TailleCtx", "-ngl", "99", "-fa", "on",
    "--split-mode", "layer",
    "--device", $Peripheriques,
    "--fit", "off",
    "-v", "--log-timestamps"
)
if ($TurboQuant) {
    $arguments += @("--cache-type-k", "q8_0", "--cache-type-v", "turbo3")
}

$libelleCache = if ($TurboQuant) { "q8_0 / turbo3" } else { "défaut (sans TurboQuant)" }

Write-Host "Démarrage llama-server (6 GPU, ctx $TailleCtx) sur http://${Hote}:${Port}" -ForegroundColor Cyan
Write-Host "  Modèle    : $Modele"
Write-Host "  Devices   : $Peripheriques"
Write-Host "  Cache K/V : $libelleCache"
Write-Host "  Chat HTML : http://${Hote}:${Port}/" -ForegroundColor Green
Write-Host ""
Write-Host "Ctrl+C pour stopper." -ForegroundColor Yellow
& $ServeurLlama @arguments
