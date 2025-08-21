@echo off
cd /d "%~dp0"
echo 현재 디렉토리: %CD%
py -3.11 -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo [OK] venv ready
pause
