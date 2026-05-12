@echo off
cd /d "%~dp0"
echo Starting API Bridge (CSU DeepSeek)...

REM 检查 Python 是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM 检查依赖
python -c "import fastapi, uvicorn, httpx" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Missing Python packages. Run: pip install fastapi uvicorn httpx
    pause
    exit /b 1
)

start "API Bridge" python server.py
timeout /t 3 >nul
set ANTHROPIC_BASE_URL=http://localhost:4000
set ANTHROPIC_AUTH_TOKEN=YOUR_CSU_API_KEY
set ANTHROPIC_MODEL=csu-deepseek[1m]
claude
pause
