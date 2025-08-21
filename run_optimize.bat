@echo off
cd /d "%~dp0"
echo 현재 디렉토리: %CD%
if not exist ".venv\Scripts\activate.bat" (
    echo 가상환경이 없습니다. setup_venv.bat을 먼저 실행하세요.
    pause
    exit /b 1
)
echo ===================================
echo 하이퍼파라미터 자동 최적화 시작
echo 예상 소요시간: 30-60분
echo ===================================
call .venv\Scripts\activate
python optimize_hyperparams.py
echo ===================================
echo 최적화 완료!
echo ===================================
pause
