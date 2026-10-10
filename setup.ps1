# One-step setup for Windows (PowerShell):  ./setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

Step "Checking Python (3.12 or newer)"
$py = $null
foreach ($candidate in @(@("py", "-3"), @("python"), @("python3"))) {
    if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
    $extra = @($candidate | Select-Object -Skip 1)
    $ok = & $candidate[0] @extra -c "import sys; print(sys.version_info >= (3, 12))" 2>$null
    if ($ok -eq "True") { $py = $candidate; break }
}
if (-not $py) { throw "Python 3.12 or newer was not found. Install it from python.org and re-run." }
$pyExtra = @($py | Select-Object -Skip 1)

Step "Creating virtual environment (.venv)"
if (-not (Test-Path ".venv")) { & $py[0] @pyExtra -m venv .venv }
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Step "Installing dependencies"
& $venvPy -m pip install --upgrade pip --quiet
$deps = & $venvPy -m scripts.setup_env deps
& $venvPy -m pip install $deps

Step "Configuring your API keys"
& $venvPy -m scripts.setup_env env

Step "Applying database migrations"
& $venvPy -m alembic upgrade head

Step "Seeding the demo business"
& $venvPy -m scripts.seed_demo

if ((Test-Path "frontend\package.json") -and (Get-Command npm -ErrorAction SilentlyContinue)) {
    Step "Installing dashboard packages"
    Push-Location frontend
    npm install
    Pop-Location
}

Write-Host "`nDone. Start the server with:" -ForegroundColor Green
Write-Host "  .venv\Scripts\activate"
Write-Host "  python -m backend.app.serve"
Write-Host "Then open http://127.0.0.1:8000"
