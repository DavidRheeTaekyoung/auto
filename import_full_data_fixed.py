"""
전체 10년 데이터를 DB로 import (수정된 버전)
"""
import pandas as pd
import os
import duckdb
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage

def import_full_data_fixed():
    """전체 10년 데이터를 DB로 import (데이터 손실 방지)"""
    
    # 설정 로드
    cfg = load_config()
    log = setup_logger(level="INFO")
    
    log.info("=== 전체 10년 데이터 import 시작 (수정된 버전) ===")
    
    # CSV 파일 경로
    csv_path = "data/BTCUSDT_5year_1min_klines.csv"
    
    if not os.path.exists(csv_path):
        log.error(f"CSV 파일이 없습니다: {csv_path}")
        return
    
    # CSV 데이터 로드
    log.info("CSV 데이터 로드 중...")
    try:
        data = pd.read_csv(csv_path)
        log.info(f"CSV 로드 완료: {len(data):,}행")
        log.info(f"기간: {data['open_time'].min()} ~ {data['open_time'].max()}")
    except Exception as e:
        log.error(f"CSV 로드 실패: {e}")
        return
    
    # 필요한 컬럼만 선택
    required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    data = data[required_columns].copy()
    
    # 데이터 타입 변환
    log.info("데이터 타입 변환 중...")
    
    # OHLCV를 숫자로 변환
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 시간을 datetime으로 변환 (이미 datetime 형태)
    data['open_time'] = pd.to_datetime(data['open_time'], utc=True)
    
    # NaN 제거
    data = data.dropna()
    
    log.info(f"데이터 타입 변환 완료: {len(data):,}행")
    
    # DB 초기화 및 데이터 저장
    log.info("DB 초기화 및 데이터 저장 중...")
    
    # 기존 테이블 삭제 (있다면)
    try:
        temp_conn = duckdb.connect(cfg["env"]["DUCKDB_PATH"])
        temp_conn.execute("DROP TABLE IF EXISTS ohlcv")
        temp_conn.close()
        log.info("기존 테이블 삭제 완료")
    except Exception as e:
        log.warning(f"테이블 삭제 실패 (무시): {e}")
    
    # 새 DataStorage 인스턴스 생성 (테이블 자동 생성)
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    log.info("새 테이블 생성 완료")
    
    # 데이터 저장 (청크 단위로, 직접 SQL 사용)
    chunk_size = 100000
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    log.info(f"총 {total_chunks}개 청크로 나누어 저장...")
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        
        try:
            # 청크 데이터에 symbol과 timeframe 추가
            chunk_with_meta = chunk.copy()
            chunk_with_meta['symbol'] = 'BTCUSDT'
            chunk_with_meta['timeframe'] = '1m'
            
            # 컬럼 순서를 테이블 스키마에 맞춤
            chunk_with_meta = chunk_with_meta[['symbol', 'timeframe', 'open_time', 'open', 'high', 'low', 'close', 'volume']]
            
            # 직접 SQL로 삽입 (upsert 대신)
            db.conn.register("chunk_df", chunk_with_meta)
            db.conn.execute("INSERT INTO ohlcv SELECT * FROM chunk_df")
            
            log.info(f"청크 {chunk_num}/{total_chunks} 저장 완료: {len(chunk):,}행")
            
        except Exception as e:
            log.error(f"청크 {chunk_num} 저장 실패: {e}")
            return
    
    # 최종 확인
    log.info("최종 데이터 확인 중...")
    try:
        final_data = db.load("BTCUSDT", "1m")
        log.info(f"최종 DB 데이터: {len(final_data):,}행")
        log.info(f"최종 기간: {final_data['open_time'].min()} ~ {final_data['open_time'].max()}")
        
        if len(final_data) > 0:
            log.info("✅ 전체 데이터 import 성공!")
            
            # 데이터 품질 확인
            log.info(f"데이터 품질 확인:")
            log.info(f"  - 시작 시간: {final_data['open_time'].min()}")
            log.info(f"  - 종료 시간: {final_data['open_time'].max()}")
            log.info(f"  - 총 행 수: {len(final_data):,}")
            log.info(f"  - 가격 범위: ${final_data['close'].min():.2f} ~ ${final_data['close'].max():.2f}")
            
        else:
            log.error("❌ DB에 데이터가 없습니다!")
            
    except Exception as e:
        log.error(f"최종 확인 실패: {e}")

if __name__ == "__main__":
    import_full_data_fixed()
