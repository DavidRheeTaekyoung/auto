import os, yaml
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    
    # 현재 스크립트 위치 기준으로 상대 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    config_path = os.path.join(project_root, "configs", "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 기본 경로들을 상대 경로로 설정
    default_db_path = os.path.join(project_root, "data", "ohlcv.duckdb")
    default_log_dir = os.path.join(project_root, "logs")
    
    cfg["env"] = {
        "DUCKDB_PATH": os.getenv("DUCKDB_PATH", default_db_path),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "TIMEZONE": os.getenv("TIMEZONE", "UTC"),
        "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY", ""),
        "BINANCE_SECRET_KEY": os.getenv("BINANCE_SECRET_KEY", ""),
        "TRADING_MODE": os.getenv("TRADING_MODE", "SIM"),  # SIM or LIVE
    }
    return cfg
