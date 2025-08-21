import pandas as pd
import os
from src.processors.data_storage import DataStorage

def import_1m_data_only():
    """1분봉 데이터만 먼저 가져오기"""
    print("=== 1분봉 데이터만 가져오기 ===")
    
    # 데이터베이스 경로
    db_path = "C:/automation/BTC/data/ohlcv.duckdb"
    store = DataStorage(db_path)
    
    print(f"데이터베이스 경로: {db_path}")
    
    # 1분봉 CSV 파일들만 처리
    csv_files = [
        "BTCUSDT_1year_1min_klines.csv",
        "BTCUSDT_5year_1min_klines.csv",
    ]
    
    total_rows = 0
    
    for filename in csv_files:
        file_path = os.path.join("data", filename)
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없습니다: {filename}")
            continue
            
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"📁 CSV 파일 로드 중: {filename} ({file_size:.1f} MB)")
        
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            print(f"  원본 컬럼: {list(df.columns)}")
            print(f"  데이터 수: {len(df):,}행")
            
            # 필요한 컬럼만 선택
            df_clean = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # open_time을 datetime으로 변환
            df_clean['open_time'] = pd.to_datetime(df_clean['open_time'], utc=True)
            
            # 숫자 컬럼들을 float으로 변환
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # NaN 값 제거
            df_clean = df_clean.dropna()
            
            print(f"  정제된 데이터 수: {len(df_clean):,}행")
            print(f"  시간 범위: {df_clean['open_time'].min()} ~ {df_clean['open_time'].max()}")
            
            # 데이터베이스에 삽입
            print(f"  데이터베이스에 삽입 중...")
            n_rows = store.upsert_ohlcv("BTC/USDT", "1m", df_clean)
            print(f"  ✅ 성공: {n_rows:,}행 삽입/업데이트")
            total_rows += n_rows
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print(f"🎉 1분봉 데이터 처리 완료!")
    print(f"총 삽입된 행 수: {total_rows:,}")
    
    # 데이터베이스 상태 확인
    print("\n=== 데이터베이스 상태 확인 ===")
    try:
        btc_1m = store.load("BTC/USDT", "1m")
        print(f"BTC/USDT 1분봉: {len(btc_1m):,}행")
        if not btc_1m.empty:
            print(f"  시간 범위: {btc_1m['open_time'].min()} ~ {btc_1m['open_time'].max()}")
            
    except Exception as e:
        print(f"데이터베이스 상태 확인 중 오류: {e}")

if __name__ == "__main__":
    import_1m_data_only()
