# -*- coding: utf-8 -*-
"""
🚀 확장 1단계 최적화: EMA + 가격변화 파라미터 (10배 확장)
더 정밀한 최적화를 위한 확장된 파라미터 범위
"""

import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass
from src.utils.config import load_config
from src.processors.data_storage import DataStorage

# =========================
# 🚀 확장: 10배 더 많은 파라미터 범위
# =========================
PARAM_RANGES_EXPANDED = {
    'ema_fast': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],  # 15개
    'ema_slow': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],  # 20개
    'price_change_period': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25],  # 15개
    'price_change_threshold': [0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],  # 15개
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

def signal_engine_expanded(df: pd.DataFrame, ema_fast=9, ema_slow=21, 
                          price_change_period=5, price_change_threshold=0.002):
    """확장: 기본 신호 엔진"""
    x = df.copy()
    x[f"ema{ema_fast}"] = ema(x["close"], ema_fast)
    x[f"ema{ema_slow}"] = ema(x["close"], ema_slow)  # 원래 방식으로 복원
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

def apply_trailing_expanded(pos: Position, price_now: float):
    """확장: 기본 트레일링 스탑"""
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

def stop_hit_expanded(pos: Position, price_now: float):
    """확장: 기본 손절"""
    if pos.side == 1:
        return price_now <= pos.entry * (1 - SL_PCT)
    else:
        return price_now >= pos.entry * (1 + SL_PCT)

def backtest_expanded(df: pd.DataFrame, **params):
    """확장: 기본 백테스트"""
    try:
        x = signal_engine_expanded(df, **params)
        
        if len(x) < 100:  # 데이터 부족
            return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100}
        
        eq = 10000.0
        equity = []
        trades = []
        pos = None
        
        # 빠른 처리를 위해 5분마다 샘플링 (10분보다 더 자주)
        x_sampled = x.iloc[::5]  # 5분마다 1개씩만 선택
        
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
                if stop_hit_expanded(pos, px) or apply_trailing_expanded(pos, px):
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
        
        # Sharpe 계산 (5분 간격이므로 12배)
        eq_series = pd.Series(equity)
        returns = eq_series.pct_change().fillna(0)
        sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(365 * 24 * 12)
        
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

def optimize_stage1_expanded():
    """🚀 확장 1단계: EMA + 가격변화 파라미터 최적화 (10배 확장)"""
    print("=== 🚀 확장 1단계 최적화 시작: EMA + 가격변화 (10배 확장) ===")
    
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
    
    # 🚀 확장: 최근 6개월 사용 (약 260,000행)
    recent_data = df.iloc[-int(len(df) * 0.05):]  # 최근 5%만 사용
    train_size = int(len(recent_data) * 0.7)
    train_data = recent_data.iloc[:train_size]
    test_data = recent_data.iloc[train_size:]
    
    print(f"전체 데이터: {len(df):,}행")
    print(f"🚀 확장 최적화용 데이터: {len(recent_data):,}행 (최근 6개월)")
    print(f"훈련 데이터: {len(train_data):,}행")
    print(f"검증 데이터: {len(test_data):,}행")
    
    # 파라미터 조합 생성
    param_combinations = list(product(
        PARAM_RANGES_EXPANDED['ema_fast'],
        PARAM_RANGES_EXPANDED['ema_slow'],
        PARAM_RANGES_EXPANDED['price_change_period'],
        PARAM_RANGES_EXPANDED['price_change_threshold']
    ))
    
    # EMA 조건 필터링 (fast < slow)
    valid_combinations = [
        (fast, slow, period, threshold) 
        for fast, slow, period, threshold in param_combinations 
        if fast < slow
    ]
    
    print(f"🚀 확장 테스트할 파라미터 조합: {len(valid_combinations)}개")
    print(f"🚀 확장 예상 실행 시간: 약 {len(valid_combinations) * 0.2 / 60:.1f}분")
    
    # 최적화 실행
    best_score = -999
    best_params = None
    results = []
    
    for i, (fast, slow, period, threshold) in enumerate(valid_combinations):
        if i % 20 == 0:  # 진행률 표시 간격 조정
            print(f"진행률: {i+1}/{len(valid_combinations)} ({(i+1)/len(valid_combinations)*100:.1f}%)")
        
        params = {
            'ema_fast': fast,
            'ema_slow': slow,
            'price_change_period': period,
            'price_change_threshold': threshold
        }
        
        # 훈련 데이터로 백테스트
        result = backtest_expanded(train_data, **params)
        result['params'] = params
        results.append(result)
        
        # 최고 점수 업데이트
        if result['score'] > best_score:
            best_score = result['score']
            best_params = params
            print(f"🔥 새로운 최고점수: {best_score:.2f}")
            print(f"   파라미터: EMA={fast}/{slow}, 기간={period}분, 임계값={threshold:.4f}")
            print(f"   수익률: {result['return']:+.2f}%, 승률: {result['win_rate']:.1f}%")
    
    # 결과 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    top_20 = results[:20]  # TOP 20으로 확장
    
    print("\n" + "="*80)
    print("🏆 확장 1단계 최적화 결과 - TOP 20")
    print("="*80)
    
    for i, result in enumerate(top_20):
        params = result['params']
        print(f"{i+1:2d}. Score: {result['score']:7.2f} | "
              f"수익률: {result['return']:+6.2f}% | "
              f"Sharpe: {result['sharpe']:6.2f} | "
              f"승률: {result['win_rate']:5.1f}% | "
              f"거래: {result['trades']:3d}건")
        print(f"     EMA: {params['ema_fast']:2d}/{params['ema_slow']:2d}, "
              f"가격변화: {params['price_change_period']:2d}분/{params['price_change_threshold']:.4f}")
        print()
    
    # 최적 파라미터로 검증 데이터 테스트
    print("="*80)
    print("🔍 확장 1단계 최적 파라미터 검증 (Out-of-Sample)")
    print("="*80)
    
    if best_params is not None:
        validation_result = backtest_expanded(test_data, **best_params)
        print(f"최적 파라미터: {best_params}")
        print(f"검증 결과:")
        print(f"  수익률: {validation_result['return']:+.2f}%")
        print(f"  Sharpe: {validation_result['sharpe']:.2f}")
        print(f"  승률: {validation_result['win_rate']:.1f}%")
        print(f"  Max DD: {validation_result['max_dd']:+.1f}%")
        print(f"  거래 수: {validation_result['trades']}건")
        
        # 2단계 최적화를 위한 결과 저장
        import json
        stage1_expanded_result = {
            'best_params': best_params,
            'validation_result': validation_result,
            'top_20': [{'params': r['params'], 'score': r['score'], 'return': r['return'], 'sharpe': r['sharpe'], 'win_rate': r['win_rate'], 'trades': r['trades']} for r in top_20]
        }
        
        with open('stage1_expanded_result.json', 'w') as f:
            json.dump(stage1_expanded_result, f, indent=2)
        
        print(f"\n✅ 확장 1단계 결과가 'stage1_expanded_result.json'에 저장되었습니다")
        print(f"🚀 다음 단계: 2단계 RSI + 볼린저밴드 최적화")
        
        # 추가 분석: 상위 5개 파라미터의 공통점 분석
        print(f"\n📊 상위 5개 파라미터 분석:")
        top_5 = results[:5]
        ema_fast_values = [r['params']['ema_fast'] for r in top_5]
        ema_slow_values = [r['params']['ema_slow'] for r in top_5]
        period_values = [r['params']['price_change_period'] for r in top_5]
        threshold_values = [r['params']['price_change_threshold'] for r in top_5]
        
        print(f"  EMA Fast 범위: {min(ema_fast_values)} ~ {max(ema_fast_values)}")
        print(f"  EMA Slow 범위: {min(ema_slow_values)} ~ {max(ema_slow_values)}")
        print(f"  가격변화 기간 범위: {min(period_values)} ~ {max(period_values)}분")
        print(f"  가격변화 임계값 범위: {min(threshold_values):.4f} ~ {max(threshold_values):.4f}")
        
    else:
        print("❌ 최적 파라미터를 찾지 못했습니다")
    
    return best_params, validation_result

if __name__ == "__main__":
    optimize_stage1_expanded()
