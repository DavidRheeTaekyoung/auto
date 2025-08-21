from src.processors.data_storage import DataStorage
from src.utils.config import load_config
import pandas as pd

def check_data():
    # 설정 로드
    cfg = load_config()
    
    # DB 데이터 확인
    print("=== DB 데이터 상태 ===")
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    try:
        raw = db.load("BTCUSDT", "1m")
        print(f"DB 데이터: {len(raw):,}행")
        print(f"기간: {raw['open_time'].min()} ~ {raw['open_time'].max()}")
    except Exception as e:
        print(f"DB 오류: {e}")
    
    # CSV 파일 확인
    print("\n=== CSV 파일 상태 ===")
    csv_path = "data/BTCUSDT_5year_1min_klines.csv"
    try:
        csv_data = pd.read_csv(csv_path)
        print(f"CSV 데이터: {len(csv_data):,}행")
        print(f"컬럼: {list(csv_data.columns)}")
        if 'open_time' in csv_data.columns:
            print(f"기간: {csv_data['open_time'].min()} ~ {csv_data['open_time'].max()}")
    except Exception as e:
        print(f"CSV 오류: {e}")

if __name__ == "__main__":
    check_data()
