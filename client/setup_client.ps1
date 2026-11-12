$ErrorActionPreference = "Stop"
$venv = Join-Path $PSScriptRoot ".venv"

if (-not (Test-Path $venv)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venv
    } else {
        & python -m venv $venv
    }
}

$pip = Join-Path $venv "Scripts\pip.exe"
& $pip install --upgrade pip
& $pip install -r (Join-Path $PSScriptRoot "requirements.txt")

$cfg = Join-Path $PSScriptRoot "config.json"
$example = Join-Path $PSScriptRoot "config.example.json"
if (-not (Test-Path $cfg) -and (Test-Path $example)) {
    Copy-Item $example $cfg
    Write-Host "Created config.json from config.example.json. Set your server_url!" -ForegroundColor Yellow
}

Write-Host "GameCopilot AI client ready." -ForegroundColor Green

