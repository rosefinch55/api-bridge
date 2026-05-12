@echo off
cd /d "%~dp0"
set ANTHROPIC_BASE_URL=http://localhost:4000
set ANTHROPIC_MODEL=csu-deepseek
litellm --config config.yaml --port 4000
pause
