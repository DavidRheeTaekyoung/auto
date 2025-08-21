@echo off
cd /d "%~dp0"
echo 현재 디렉토리: %CD%
if not exist ".venv\Scripts\activate.bat" (
    echo 가상환경이 없습니다. setup_venv.bat을 먼저 실행하세요.
    pause
    exit /b 1
)
call .venv\Scripts\activate
python train.py
pause
