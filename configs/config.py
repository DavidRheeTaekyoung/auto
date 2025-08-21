import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API Keys (환경변수에서 로드)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', '')
COINBASE_SECRET_KEY = os.getenv('COINBASE_SECRET_KEY', '')
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
KRAKEN_SECRET_KEY = os.getenv('KRAKEN_SECRET_KEY', '')

# Database
DB_TYPE = os.getenv('DB_TYPE', 'duckdb')
DUCKDB_PATH = os.getenv('DUCKDB_PATH', './data/ohlcv.duckdb')
TIMESCALE_HOST = os.getenv('TIMESCALE_HOST', 'localhost')
TIMESCALE_PORT = int(os.getenv('TIMESCALE_PORT', '5432'))
TIMESCALE_DB = os.getenv('TIMESCALE_DB', 'crypto')
TIMESCALE_USER = os.getenv('TIMESCALE_USER', 'postgres')
TIMESCALE_PASSWORD = os.getenv('TIMESCALE_PASSWORD', 'password')

# 설정
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
TIMEZONE = os.getenv('TIMEZONE', 'UTC')
TRADING_MODE = os.getenv('TRADING_MODE', 'SIM')  # SIM 또는 LIVE

# 거래 설정
TRADING_CONFIG = {
    'exchanges': ['binance', 'coinbase', 'kraken'],
    'symbols': ['BTC/USD', 'ETH/USD'],
    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'lookback_days': 7
}

# 모델 설정
MODEL_CONFIG = {
    'regime_detector': {
        'n_states': 5,
        'features': ['returns', 'volatility', 'skewness', 'kurtosis', 'volume_z', 'trend_strength']
    },
    'nhits': {
        'input_length': 512,
        'horizons': {'1h': 60, '4h': 240, '24h': 1440, '1w': 10080},
        'quantiles': [0.1, 0.5, 0.9],
        'learning_rate': 0.0003,
        'batch_size': 256
    }
}

print("=== 설정 파일 로드 완료 ===")
print(f"거래 모드: {TRADING_MODE}")
print(f"바이낸스 API 키: {'설정됨' if BINANCE_API_KEY else '설정되지 않음'}")
print(f"데이터베이스: {DUCKDB_PATH}")
print("=" * 30)
