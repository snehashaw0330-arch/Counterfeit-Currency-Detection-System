# Counterfeit Currency Detection — one-command local launch (Windows).
#
# Opens the backend (FastAPI/uvicorn) and the frontend (Next.js) each
# in their own PowerShell window. Run from the project root:
#
#     powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
#
# Backend:  http://127.0.0.1:8000  (Swagger: /docs ; debug: POST /diagnose)
# Frontend: http://localhost:3000

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $root "venv\Scripts\Activate.ps1"))) {
    Write-Error "venv not found at venv\. Create it: python -m venv venv ; then pip install -r requirements.txt"
    exit 1
}

Write-Host "Starting backend (uvicorn) on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; .\venv\Scripts\Activate.ps1; uvicorn backend.main:app --host 127.0.0.1 --port 8000"
)

Write-Host "Starting frontend (npm run dev) on http://localhost:3000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host ""
Write-Host "Both started in separate windows." -ForegroundColor Cyan
Write-Host "  Backend : http://127.0.0.1:8000/docs"
Write-Host "  Frontend: http://localhost:3000"
Write-Host "First backend launch downloads ~64 MB of EasyOCR weights (one-time)."
