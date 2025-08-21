import os

# ⚠️ 보안 주의사항 ⚠️
# 이 파일에 실제 API 키를 직접 입력하지 마세요!
# 환경변수나 별도 파일을 사용하세요.

print("=== 비트코인 거래 시스템 환경변수 설정 ===")
print("실거래 전에 다음 설정을 확인하세요:")
print()

# 환경변수 설정 (실행 전 수정 필요)
# 방법 1: 이 파일에서 직접 설정 (보안상 권장하지 않음)
# os.environ["BINANCE_API_KEY"] = "your_api_key_here"
# os.environ["BINANCE_SECRET_KEY"] = "your_secret_key_here"

# 방법 2: 환경변수에서 로드 (권장)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# 거래 모드 설정
TRADING_MODE = os.getenv("TRADING_MODE", "SIM")  # SIM 또는 LIVE
TIMEZONE = os.getenv("TIMEZONE", "UTC")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "C:/automation/BTC/data/ohlcv.duckdb")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

print(f"거래 모드: {TRADING_MODE}")
print(f"바이낸스 API 키: {'설정됨' if BINANCE_API_KEY else '설정되지 않음'}")
print(f"데이터베이스 경로: {DUCKDB_PATH}")
print(f"로그 레벨: {LOG_LEVEL}")
print()

if TRADING_MODE == "LIVE":
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        print("❌ 실거래 모드이지만 API 키가 설정되지 않았습니다!")
        print("   config.py에서 API 키를 설정하거나 환경변수를 설정하세요.")
        print("   또는 TRADING_MODE=SIM으로 변경하세요.")
    else:
        print("✅ 실거래 모드 - API 키 설정됨")
        print("   ⚠️ 주의: 실제 거래가 발생합니다!")
else:
    print("✅ 모의거래 모드 - 안전하게 테스트 가능")

print()
print("=== 사용법 ===")
print("1. 모의거래: TRADING_MODE=SIM (기본값)")
print("2. 실거래: TRADING_MODE=LIVE + API 키 설정")
print("3. 환경변수 설정 예시:")
print("   set BINANCE_API_KEY=your_key_here")
print("   set BINANCE_SECRET_KEY=your_secret_here")
print("   set TRADING_MODE=LIVE")
print("=" * 50)
