# Compiler llama-server (Ninja + vcvars64)
param(
    [string]$CheminDepot = (Join-Path $env:USERPROFILE "Documents\llama-cpp-turboquant"),
    [int]$NbJobs = 8
)

$ErrorActionPreference = "Stop"
$CheminDepot = (Resolve-Path -LiteralPath $CheminDepot).Path

$candidatsVcvars = @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $candidatsVcvars | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) {
    Write-Error "vcvars64.bat introuvable. Installez la charge VS 2022 C++."
}

$commande = @(
    "call `"$vcvars`"",
    "cd /d `"$CheminDepot`"",
    "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF",
    "cmake --build build --target llama-server -j $NbJobs"
) -join " && "

Write-Host "Compilation dans $CheminDepot ..." -ForegroundColor Cyan
cmd /c $commande
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$binaire = Join-Path $CheminDepot "build\bin\llama-server.exe"
if (-not (Test-Path $binaire)) { Write-Error "Binaire introuvable : $binaire" }
Write-Host "OK : $binaire" -ForegroundColor Green
