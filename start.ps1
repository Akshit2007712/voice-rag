# Voice RAG - Start Both Backend + Frontend
# Run from the project root: .\start.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Voice RAG - Starting Backend + Frontend" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"

# Start Backend in a new PowerShell window
Write-Host "[1/2] Launching FastAPI Backend (port 8000)..." -ForegroundColor Yellow
$backendDir = Join-Path $PSScriptRoot "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendDir'; `$env:PYTHONIOENCODING='utf-8'; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info" -WindowStyle Normal

# Wait for backend to boot
Write-Host "Waiting for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 4

# Show URLs
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Backend  : http://localhost:8000" -ForegroundColor Green
Write-Host "  Frontend : http://localhost:3005" -ForegroundColor Green
Write-Host "  API Docs : http://localhost:8000/docs" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Start Frontend in current window
Write-Host "[2/2] Launching Next.js Frontend (port 3005)..." -ForegroundColor Yellow
Set-Location $PSScriptRoot
npm run dev -- -p 3005
