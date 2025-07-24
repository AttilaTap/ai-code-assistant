@echo off
set /p INPUT="Enter local folder path or Git repo URL: "
call venv\Scripts\activate.bat

REM Check if it's a Git URL (starts with https), else treat as folder
echo %INPUT% | findstr /b "https://" >nul
if %errorlevel%==0 (
    python run.py --git %INPUT%
) else (
    python run.py --codebase %INPUT%
)

pause
