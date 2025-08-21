# -*- coding: utf-8 -*-
"""
🚀 초고속 1단계 최적화: EMA + 가격변화 파라미터
1-2분 내 완료를 위한 극소량 파라미터만 최적화
"""

import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass
from src.utils.config import load_config
from src.processors.data_storage import DataStorage

# =========================
# 🚀 초고속: 극소량 파라미터만 최적화
# =========================
PARAM_RANGES_FAST = {
    'ema_fast': [5, 9, 12],           # 3개만
    'ema_slow': [18, 21, 26],         # 3개만
    'price_change_period': [3, 5, 10],  # 3개만
    'price_change_threshold': [0.001, 0.002, 0.003],  # 3개만
}

# 고정 파라미터
LEV = 5            # 레버리지 5배 고정
SL_PCT = 0.003     # 손절 0.3% 고정
TP_ARM_PCT = 0.01  # 트레일링 활성화 1% 고정
GIVEBACK = 0.20    # 이익 20% 반납 고정
NOTIONAL_USD = 10_000
FEE_BP = 6
SLIP_BP = 5

# =========================
# 유틸
# =========================
def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()

@dataclass
class Position:
    side: int
    entry: float
    qty: float
    peak: float
    armed: bool

def signal_engine_fast(df: pd.DataFrame, ema_fast=9, ema_slow=21, 
                       price_change_period=5, price_change_threshold=0.002):
    """초고속: 기본 신호 엔진"""
    x = df.copy()
    x[f"ema{ema_fast}"] = ema(x["close"], ema_fast)
    x[f"ema{ema_slow}"] = ema(x[f"ema{ema_fast}"], ema_slow)  # 빠른 EMA 위에 느린 EMA
    x["cr"] = np.sign(x[f"ema{ema_fast}"] - x[f"ema{ema_slow}"])
    x["cr_prev"] = x["cr"].shift(1)
    
    # 가격 변화율 계산
    x["price_change"] = x["close"].pct_change(price_change_period)
    
    # 신호 생성 (EMA 크로스 + 가격 변화)
    sig = np.where(
        (x["cr_prev"] <= 0) & (x["cr"] > 0) & (x["price_change"] > price_change_threshold), 1,
        np.where(
            (x["cr_prev"] >= 0) & (x["cr"] < 0) & (x["price_change"] < -price_change_threshold), -1, 0
        )
    )
    x["sig"] = sig
    return x[["open", "high", "low", "close", "volume", f"ema{ema_fast}", f"ema{ema_slow}", "price_change", "sig"]].dropna()

def apply_trailing_fast(pos: Position, price_now: float):
    """초고속: 기본 트레일링 스탑"""
    if pos.side == 1:
        gain = (price_now / pos.entry) - 1.0
        if not pos.armed and gain >= TP_ARM_PCT:
            pos.armed = True
            pos.peak = max(pos.peak, price_now)
        if pos.armed:
            pos.peak = max(pos.peak, price_now)
            trigger = pos.entry * (1 + (pos.peak / pos.entry - 1) * (1 - GIVEBACK))
            if price_now <= trigger:
                return True
        return False
    else:
        gain = (pos.entry / price_now) - 1.0
        if not pos.armed and gain >= TP_ARM_PCT:
            pos.armed = True
            pos.peak = min(pos.peak, price_now)
        if pos.armed:
            pos.peak = min(pos.peak, price_now)
            trigger = pos.entry * (1 - (1 - pos.peak / pos.entry) * (1 - GIVEBACK))
            if price_now >= trigger:
                return True
        return False

def stop_hit_fast(pos: Position, price_now: float):
    """초고속: 기본 손절"""
    if pos.side == 1:
        return price_now <= pos.entry * (1 - SL_PCT)
    else:
        return price_now >= pos.entry * (1 + SL_PCT)

def backtest_fast(df: pd.DataFrame, **params):
    """초고속: 기본 백테스트"""
    try:
        x = signal_engine_fast(df, **params)
        
        if len(x) < 50:  # 데이터 부족 (더 작게)
            return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100}
        
        eq = 10000.0
        equity = []
        trades = []
        pos = None
        
        # 더 빠른 처리를 위해 10분마다 샘플링
        x_sampled = x.iloc[::10]  # 10분마다 1개씩만 선택
        
        for t, row in x_sampled.iterrows():
            px = float(row["close"])
            sig = int(row["sig"])
            
            # 진입
            if pos is None and sig != 0:
                qty = (NOTIONAL_USD * LEV) / px
                pos = Position(side=sig, entry=px, qty=qty, peak=px, armed=False)
                eq -= (qty * px) * (FEE_BP / 2 / 1e4 + SLIP_BP / 2 / 1e4)
            
            # 청산
            if pos is not None:
                if stop_hit_fast(pos, px) or apply_trailing_fast(pos, px):
                    pnl = pos.qty * (px - pos.entry) * pos.side
                    cost = (pos.qty * px) * (FEE_BP / 2 / 1e4 + SLIP_BP / 2 / 1e4)
                    eq += pnl - cost
                    trades.append({
                        "time": t, 
                        "side": "LONG" if pos.side == 1 else "SHORT",
                        "entry": pos.entry, 
                        "exit": px, 
                        "pnl": pnl, 
                        "eq": eq
                    })
                    pos = None
            
            equity.append(eq if pos is None else eq + pos.qty * (px - pos.entry) * pos.side)
        
        # 성과 계산
        if len(trades) == 0:
            return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100}
        
        final_return = (eq - 10000) / 10000
        
        # Sharpe 계산 (간단화)
        eq_series = pd.Series(equity)
        returns = eq_series.pct_change().fillna(0)
        sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(365 * 24 * 6)  # 10분 간격이므로 6배
        
        # Max Drawdown
        dd = (eq_series / eq_series.cummax() - 1).min()
        
        # 승률
        trade_returns = [(t['pnl'] / (t['entry'] * LEV / 10000)) for t in trades]
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
        
        # 복합 점수 (수익률 + Sharpe + 승률 - Max DD)
        score = final_return * 100 + sharpe * 10 + win_rate * 50 - abs(dd) * 100
        
        return {
            'score': score,
            'trades': len(trades),
            'return': final_return * 100,
            'sharpe': sharpe,
            'max_dd': dd * 100,
            'win_rate': win_rate * 100,
            'final_eq': eq
        }
        
    except Exception as e:
        print(f"백테스트 오류: {e}")
        return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100}

def optimize_stage1_fast():
    """🚀 초고속 1단계: EMA + 가격변화 파라미터 최적화"""
    print("=== 🚀 초고속 1단계 최적화 시작: EMA + 가격변화 ===")
    
    # 10년 데이터 로드
    cfg = load_config()
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    symbol_db = "BTCUSDT"
    raw_data = db.load(symbol_db, "1m")
    
    if len(raw_data) == 0:
        print("❌ 데이터가 없습니다")
        return
    
    print(f"✅ 전체 데이터 로드: {len(raw_data):,}행")
    print(f"기간: {raw_data['open_time'].min()} ~ {raw_data['open_time'].max()}")
    
    # 데이터 형태 변환
    df = raw_data.copy()
    df = df.set_index(pd.to_datetime(df['open_time']))
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # 🚀 초고속: 최근 3개월만 사용 (약 130,000행)
    recent_data = df.iloc[-int(len(df) * 0.025):]  # 최근 2.5%만 사용
    train_size = int(len(recent_data) * 0.7)
    train_data = recent_data.iloc[:train_size]
    test_data = recent_data.iloc[train_size:]
    
    print(f"전체 데이터: {len(df):,}행")
    print(f"🚀 최적화용 데이터: {len(recent_data):,}행 (최근 3개월)")
    print(f"훈련 데이터: {len(train_data):,}행")
    print(f"검증 데이터: {len(test_data):,}행")
    
    # 파라미터 조합 생성
    param_combinations = list(product(
        PARAM_RANGES_FAST['ema_fast'],
        PARAM_RANGES_FAST['ema_slow'],
        PARAM_RANGES_FAST['price_change_period'],
        PARAM_RANGES_FAST['price_change_threshold']
    ))
    
    # EMA 조건 필터링 (fast < slow)
    valid_combinations = [
        (fast, slow, period, threshold) 
        for fast, slow, period, threshold in param_combinations 
        if fast < slow
    ]
    
    print(f"🚀 테스트할 파라미터 조합: {len(valid_combinations)}개")
    print(f"🚀 예상 실행 시간: 약 {len(valid_combinations) * 0.1 / 60:.1f}분 (초고속!)")
    
    # 최적화 실행
    best_score = -999
    best_params = None
    results = []
    
    for i, (fast, slow, period, threshold) in enumerate(valid_combinations):
        if i % 5 == 0:  # 더 자주 진행률 표시
            print(f"진행률: {i+1}/{len(valid_combinations)} ({(i+1)/len(valid_combinations)*100:.1f}%)")
        
        params = {
            'ema_fast': fast,
            'ema_slow': slow,
            'price_change_period': period,
            'price_change_threshold': threshold
        }
        
        # 훈련 데이터로 백테스트
        result = backtest_fast(train_data, **params)
        result['params'] = params
        results.append(result)
        
        # 최고 점수 업데이트
        if result['score'] > best_score:
            best_score = result['score']
            best_params = params
            print(f"🔥 새로운 최고점수: {best_score:.2f}")
            print(f"   파라미터: EMA={fast}/{slow}, 기간={period}분, 임계값={threshold:.3f}")
            print(f"   수익률: {result['return']:+.2f}%, 승률: {result['win_rate']:.1f}%")
    
    # 결과 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    top_10 = results[:10]
    
    print("\n" + "="*80)
    print("🏆 초고속 1단계 최적화 결과 - TOP 10")
    print("="*80)
    
    for i, result in enumerate(top_10):
        params = result['params']
        print(f"{i+1:2d}. Score: {result['score']:7.2f} | "
              f"수익률: {result['return']:+6.2f}% | "
              f"Sharpe: {result['sharpe']:6.2f} | "
              f"승률: {result['win_rate']:5.1f}% | "
              f"거래: {result['trades']:3d}건")
        print(f"     EMA: {params['ema_fast']}/{params['ema_slow']}, "
              f"가격변화: {params['price_change_period']}분/{params['price_change_threshold']:.3f}")
        print()
    
    # 최적 파라미터로 검증 데이터 테스트
    print("="*80)
    print("🔍 초고속 1단계 최적 파라미터 검증 (Out-of-Sample)")
    print("="*80)
    
    if best_params is not None:
        validation_result = backtest_fast(test_data, **best_params)
        print(f"최적 파라미터: {best_params}")
        print(f"검증 결과:")
        print(f"  수익률: {validation_result['return']:+.2f}%")
        print(f"  Sharpe: {validation_result['sharpe']:.2f}")
        print(f"  승률: {validation_result['win_rate']:.1f}%")
        print(f"  Max DD: {validation_result['max_dd']:+.1f}%")
        print(f"  거래 수: {validation_result['trades']}건")
        
        # 2단계 최적화를 위한 결과 저장
        import json
        stage1_fast_result = {
            'best_params': best_params,
            'validation_result': validation_result,
            'top_10': [{'params': r['params'], 'score': r['score'], 'return': r['return'], 'sharpe': r['sharpe'], 'win_rate': r['win_rate'], 'trades': r['trades']} for r in top_10]
        }
        
        with open('stage1_fast_result.json', 'w') as f:
            json.dump(stage1_fast_result, f, indent=2)
        
        print(f"\n✅ 초고속 1단계 결과가 'stage1_fast_result.json'에 저장되었습니다")
        print(f"🚀 다음 단계: 2단계 RSI + 볼린저밴드 최적화")
        
    else:
        print("❌ 최적 파라미터를 찾지 못했습니다")
    
    return best_params, validation_result

if __name__ == "__main__":
    optimize_stage1_fast()
