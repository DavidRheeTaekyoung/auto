# -*- coding: utf-8 -*-
"""
🔍 2단계 최적화 디버깅: RSI + 볼린저밴드 조건 테스트
"""

import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.processors.data_storage import DataStorage

def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n):
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(s, n, std):
    sma = s.rolling(window=n).mean()
    std_dev = s.rolling(window=n).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower

def test_stage2_conditions():
    """2단계 조건 테스트"""
    print("=== 🔍 2단계 조건 디버깅 ===")
    
    # 데이터 로드
    cfg = load_config()
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    symbol_db = "BTCUSDT"
    raw_data = db.load(symbol_db, "1m")
    
    if len(raw_data) == 0:
        print("❌ 데이터가 없습니다")
        return
    
    # 최근 1개월 데이터만 사용
    df = raw_data.copy()
    df = df.set_index(pd.to_datetime(df['open_time']))
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    recent_data = df.iloc[-int(len(df) * 0.008):]  # 최근 0.8%만 사용 (약 1개월)
    print(f"테스트 데이터: {len(recent_data):,}행")
    
    # 1단계 파라미터 (고정)
    ema_fast = 17
    ema_slow = 21
    price_change_period = 20
    price_change_threshold = 0.0007
    
    # 2단계 파라미터 (테스트)
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    bb_period = 20
    bb_std = 2.0
    
    # 지표 계산
    x = recent_data.copy()
    x[f"ema{ema_fast}"] = ema(x["close"], ema_fast)
    x[f"ema{ema_slow}"] = ema(x["close"], ema_slow)
    x["cr"] = np.sign(x[f"ema{ema_fast}"] - x[f"ema{ema_slow}"])
    x["cr_prev"] = x["cr"].shift(1)
    x["price_change"] = x["close"].pct_change(price_change_period)
    x["rsi"] = rsi(x["close"], rsi_period)
    x["bb_upper"], x["bb_lower"] = bollinger_bands(x["close"], bb_period, bb_std)
    x["bb_position"] = (x["close"] - x["bb_lower"]) / (x["bb_upper"] - x["bb_lower"])
    
    # 조건별 신호 개수 확인
    print(f"\n📊 조건별 신호 분석:")
    
    # 1단계 조건만
    stage1_long = (x["cr_prev"] <= 0) & (x["cr"] > 0) & (x["price_change"] > price_change_threshold)
    stage1_short = (x["cr_prev"] >= 0) & (x["cr"] < 0) & (x["price_change"] < -price_change_threshold)
    
    print(f"1단계 조건만:")
    print(f"  LONG 신호: {stage1_long.sum()}건")
    print(f"  SHORT 신호: {stage1_short.sum()}건")
    
    # 2단계 조건만
    stage2_long = (x["rsi"] < rsi_oversold) & (x["bb_position"] < 0.3)
    stage2_short = (x["rsi"] > rsi_overbought) & (x["bb_position"] > 0.7)
    
    print(f"\n2단계 조건만:")
    print(f"  RSI 과매도 + BB 하단: {stage2_long.sum()}건")
    print(f"  RSI 과매수 + BB 상단: {stage2_short.sum()}건")
    
    # RSI 조건만
    rsi_long = x["rsi"] < rsi_oversold
    rsi_short = x["rsi"] > rsi_overbought
    
    print(f"\nRSI 조건만:")
    print(f"  RSI < {rsi_oversold}: {rsi_long.sum()}건")
    print(f"  RSI > {rsi_overbought}: {rsi_short.sum()}건")
    
    # BB 조건만
    bb_long = x["bb_position"] < 0.3
    bb_short = x["bb_position"] > 0.7
    
    print(f"\nBB 조건만:")
    print(f"  BB 위치 < 0.3: {bb_long.sum()}건")
    print(f"  BB 위치 > 0.7: {bb_short.sum()}건")
    
    # 통합 조건
    final_long = stage1_long & stage2_long
    final_short = stage1_short & stage2_short
    
    print(f"\n통합 조건:")
    print(f"  최종 LONG: {final_long.sum()}건")
    print(f"  최종 SHORT: {final_short.sum()}건")
    
    # 샘플 데이터 확인
    print(f"\n📊 샘플 데이터 (처음 10개):")
    sample = x[["close", f"ema{ema_fast}", f"ema{ema_slow}", "cr", "cr_prev", 
                "price_change", "rsi", "bb_position"]].head(10)
    print(sample.round(4))
    
    # RSI와 BB 분포 확인
    print(f"\n📊 RSI 분포:")
    print(f"  최소: {x['rsi'].min():.1f}")
    print(f"  최대: {x['rsi'].max():.1f}")
    print(f"  평균: {x['rsi'].mean():.1f}")
    print(f"  표준편차: {x['rsi'].std():.1f}")
    
    print(f"\n📊 BB 위치 분포:")
    print(f"  최소: {x['bb_position'].min():.3f}")
    print(f"  최대: {x['bb_position'].max():.3f}")
    print(f"  평균: {x['bb_position'].mean():.3f}")
    print(f"  표준편차: {x['bb_position'].std():.3f}")
    
    # 조건 완화 제안
    print(f"\n💡 조건 완화 제안:")
    print(f"  RSI 과매도: {rsi_oversold} → {rsi_oversold + 10} (더 관대하게)")
    print(f"  RSI 과매수: {rsi_overbought} → {rsi_overbought - 10} (더 관대하게)")
    print(f"  BB 하단: 0.3 → 0.4 (더 관대하게)")
    print(f"  BB 상단: 0.7 → 0.6 (더 관대하게)")

if __name__ == "__main__":
    test_stage2_conditions()
