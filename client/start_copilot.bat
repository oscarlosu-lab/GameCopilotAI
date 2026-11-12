@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up GameCopilot AI client virtual environment...
    powershell -ExecutionPolicy Bypass -File "%~dp0setup_client.ps1"
)
".venv\Scripts\python.exe" "%~dp0copilot_client.py"
pause

