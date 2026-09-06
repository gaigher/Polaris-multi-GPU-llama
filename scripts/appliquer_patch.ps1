# Appliquer polaris-subgroup.patch sur un clone TheTom
param(
    [string]$CheminDepot = (Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant"),
    [string]$CheminPatch = (Join-Path $PSScriptRoot "..\patches\polaris-subgroup.patch")
)

$ErrorActionPreference = "Stop"
$CheminDepot = (Resolve-Path -LiteralPath $CheminDepot).Path
$CheminPatch = (Resolve-Path -LiteralPath $CheminPatch).Path

if (-not (Test-Path (Join-Path $CheminDepot ".git"))) {
    Write-Error "Pas un dépôt git : $CheminDepot. Clonez TheTom d'abord."
}

Push-Location $CheminDepot
try {
    git apply --check $CheminPatch
    git apply $CheminPatch
    Write-Host "Patch appliqué dans $CheminDepot" -ForegroundColor Green
    git diff --stat
} finally {
    Pop-Location
}
