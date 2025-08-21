import pandas as pd
import os
import sys
from datetime import datetime

def import_csv_to_duckdb():
    """CSV 파일들을 DuckDB 데이터베이스로 가져오기"""
    print("=== CSV 데이터를 DuckDB로 가져오기 ===")
    
    # 데이터베이스 경로
    db_path = "C:/automation/BTC/data/ohlcv.duckdb"
    
    # DataStorage 클래스 import
    from src.processors.data_storage import DataStorage
    store = DataStorage(db_path)
    
    print(f"데이터베이스 경로: {db_path}")
    
    # Data directory
    data_dir = "data"
    
    # OHLCV 데이터가 포함된 CSV 파일들
    csv_files_to_process = [
        "BTCUSDT_1d_klines.csv",
        "BTCUSDT_1year_1min_klines.csv", 
        "BTCUSDT_5year_1min_klines.csv",
        "BTCUSDT_daily_klines.csv",
    ]
    
    # 실제로 존재하는 파일들만 필터링
    existing_csv_files = [f for f in csv_files_to_process if os.path.exists(os.path.join(data_dir, f))]
    
    if not existing_csv_files:
        print(f"'{data_dir}' 디렉토리에서 가져올 OHLCV CSV 파일을 찾을 수 없습니다.")
        return
    
    print(f"처리할 CSV 파일들: {existing_csv_files}")
    print()
    
    total_rows = 0
    
    for filename in existing_csv_files:
        file_path = os.path.join(data_dir, filename)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        print(f"📁 CSV 파일 로드 중: {filename} ({file_size:.1f} MB)")
        
        try:
            # CSV 파일 읽기 (헤더가 있음)
            df = pd.read_csv(file_path)
            
            print(f"  원본 컬럼: {list(df.columns)}")
            print(f"  데이터 수: {len(df):,}행")
            
            # 필요한 컬럼만 선택하고 이름 변경
            if 'open_time' in df.columns:
                # 이미 올바른 컬럼명이 있는 경우
                df_clean = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
            else:
                # 컬럼명이 다른 경우 처리
                print(f"  ⚠️ 'open_time' 컬럼을 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # open_time을 datetime으로 변환
            df_clean['open_time'] = pd.to_datetime(df_clean['open_time'], utc=True)
            
            # 숫자 컬럼들을 float으로 변환
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # NaN 값 제거
            df_clean = df_clean.dropna()
            
            # 심볼과 타임프레임 결정
            symbol_internal = "BTC/USDT"
            
            if "1min_klines" in filename:
                timeframe = "1m"
            elif "1d_klines" in filename or "daily_klines" in filename:
                timeframe = "1d"
            else:
                timeframe = "1m"  # 기본값
            
            print(f"  심볼: {symbol_internal}")
            print(f"  타임프레임: {timeframe}")
            print(f"  시간 범위: {df_clean['open_time'].min()} ~ {df_clean['open_time'].max()}")
            print(f"  정제된 데이터 수: {len(df_clean):,}행")
            
            # 데이터베이스에 삽입
            print(f"  데이터베이스에 삽입 중...")
            n_rows = store.upsert_ohlcv(symbol_internal, timeframe, df_clean)
            print(f"  ✅ 성공: {n_rows:,}행 삽입/업데이트")
            total_rows += n_rows
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print(f"🎉 모든 CSV 파일 처리 완료!")
    print(f"총 삽입된 행 수: {total_rows:,}")
    
    # 데이터베이스 상태 확인
    print("\n=== 데이터베이스 상태 확인 ===")
    try:
        btc_1m = store.load("BTC/USDT", "1m")
        btc_1d = store.load("BTC/USDT", "1d")
        
        print(f"BTC/USDT 1분봉: {len(btc_1m):,}행")
        if not btc_1m.empty:
            print(f"  시간 범위: {btc_1m['open_time'].min()} ~ {btc_1m['open_time'].max()}")
        
        print(f"BTC/USDT 1일봉: {len(btc_1d):,}행")
        if not btc_1d.empty:
            print(f"  시간 범위: {btc_1d['open_time'].min()} ~ {btc_1d['open_time'].max()}")
            
    except Exception as e:
        print(f"데이터베이스 상태 확인 중 오류: {e}")

if __name__ == "__main__":
    import_csv_to_duckdb()
