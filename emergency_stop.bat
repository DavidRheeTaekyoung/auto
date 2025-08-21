@echo off
cd /d "%~dp0"
echo 현재 디렉토리: %CD%
echo 긴급 정지 실행 중...
taskkill /FI "WINDOWTITLE eq run_live" /F 2>nul
taskkill /FI "WINDOWTITLE eq run_collect" /F 2>nul
echo Stopped live/collector loops.
pause
