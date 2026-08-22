@echo off
echo ============================================================
echo   Voice RAG - Starting Backend + Frontend
echo ============================================================
echo.

:: Set PYTHONIOENCODING for UTF-8 terminal output
set PYTHONIOENCODING=utf-8

:: Start backend in a new window
echo [1/2] Starting FastAPI Backend on port 8000...
start "Voice RAG - Backend" cmd /k "cd /d "%~dp0backend" && set PYTHONIOENCODING=utf-8 && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info"

:: Wait for backend to initialize
echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

:: Start frontend in current window
echo [2/2] Starting Next.js Frontend on port 3005...
echo.
echo ============================================================
echo  Backend: http://localhost:8000
echo  Frontend: http://localhost:3005
echo  API Docs: http://localhost:8000/docs
echo ============================================================
echo.

cd /d "%~dp0"
npm run dev -- -p 3005
