# Applique les patchs Vulkan GCN (FA LDS / MMQ proprietary) sur un clone TheTom.
# Variantes isolées pour le protocole 4 builds — voir docs/Plan Optimiser PP-TG-VRAM.md
param(
    [ValidateSet("fa", "mmq", "both", "check")]
    [string]$Variante = "check",
    [string]$CheminDepot = $(
        if (Test-Path (Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant\.git")) {
            Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant"
        } else {
            Join-Path (Split-Path $PSScriptRoot -Parent | Split-Path -Parent) "llama-cpp-turboquant-feature-turboquant-kv-cache"
        }
    ),
    [switch]$Reverse
)

$ErrorActionPreference = "Stop"
$patchDir = Join-Path $PSScriptRoot "..\patches"

$map = @{
    fa   = "gcn-fa-lds.patch"
    mmq  = "gcn-mmq-proprietary.patch"
    both = "gcn-fa-mmq-combined.patch"
}

$CheminDepot = (Resolve-Path -LiteralPath $CheminDepot).Path
if (-not (Test-Path (Join-Path $CheminDepot ".git"))) {
    Write-Error "Pas un depot git : $CheminDepot"
}

function Resolve-PatchPath([string]$Name) {
    $p = Join-Path $patchDir $Name
    return (Resolve-Path -LiteralPath $p).Path
}

Push-Location $CheminDepot
try {
    $sha = (git rev-parse --short HEAD).Trim()
    Write-Host "Depot : $CheminDepot ($sha)" -ForegroundColor Cyan

    if ($Variante -eq "check") {
        foreach ($k in @("fa", "mmq", "both")) {
            $patch = Resolve-PatchPath $map[$k]
            Write-Host "check $k -> $patch"
            git apply --check $patch
            Write-Host "  OK" -ForegroundColor Green
        }
        return
    }

    $patch = Resolve-PatchPath $map[$Variante]
    $args = @()
    if ($Reverse) { $args += "--reverse" }
    git apply --check @args $patch
    git apply @args $patch
    $action = if ($Reverse) { "retire" } else { "applique" }
    Write-Host "Patch $Variante $action ($patch)" -ForegroundColor Green
    git diff --stat -- ggml/src/ggml-vulkan/ggml-vulkan.cpp
}
finally {
    Pop-Location
}
