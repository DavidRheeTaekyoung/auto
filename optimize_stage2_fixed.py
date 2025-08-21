# -*- coding: utf-8 -*-
"""
🚀 2단계 최적화 (수정): RSI + 볼린저밴드 파라미터
1단계에서 찾은 최적 EMA 파라미터를 고정하고 RSI + BB만 최적화
"""

import pandas as pd
import numpy as np
import json
from itertools import product
from dataclasses import dataclass
from src.utils.config import load_config
from src.processors.data_storage import DataStorage

# =========================
# 🚀 2단계: RSI + 볼린저밴드 파라미터 범위 (수정)
# =========================
PARAM_RANGES_STAGE2 = {
    'rsi_period': [7, 9, 11, 14, 17, 20, 23, 26],  # 8개
    'rsi_oversold': [25, 30, 35, 40],  # 4개 (더 관대하게)
    'rsi_overbought': [60, 65, 70, 75],  # 4개 (더 관대하게)
    'bb_period': [15, 18, 21, 24, 27, 30],  # 6개
    'bb_std': [1.5, 1.8, 2.0, 2.2, 2.5, 2.8],  # 6개
    'confirmation_bars': [1, 2, 3, 5],  # 4개 (확인 바 수)
}

# 1단계에서 찾은 최적 파라미터 (고정)
STAGE1_BEST_PARAMS = {
    'ema_fast': 17,
    'ema_slow': 21,
    'price_change_period': 20,
    'price_change_threshold': 0.0007
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

@dataclass
class Position:
    side: int
    entry: float
    qty: float
    peak: float
    armed: bool

def signal_engine_stage2_fixed(df: pd.DataFrame, **params):
    """2단계 (수정): RSI + 볼린저밴드가 추가된 신호 엔진"""
    x = df.copy()
    
    # 1단계 파라미터 적용 (고정)
    ema_fast = STAGE1_BEST_PARAMS['ema_fast']
    ema_slow = STAGE1_BEST_PARAMS['ema_slow']
    price_change_period = STAGE1_BEST_PARAMS['price_change_period']
    price_change_threshold = STAGE1_BEST_PARAMS['price_change_threshold']
    
    # 2단계 파라미터
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    confirmation_bars = params.get('confirmation_bars', 2)
    
    # 1단계 지표 계산
    x[f"ema{ema_fast}"] = ema(x["close"], ema_fast)
    x[f"ema{ema_slow}"] = ema(x["close"], ema_slow)
    x["cr"] = np.sign(x[f"ema{ema_fast}"] - x[f"ema{ema_slow}"])
    x["cr_prev"] = x["cr"].shift(1)
    x["price_change"] = x["close"].pct_change(price_change_period)
    
    # 2단계 지표 계산
    x["rsi"] = rsi(x["close"], rsi_period)
    x["bb_upper"], x["bb_lower"] = bollinger_bands(x["close"], bb_period, bb_std)
    x["bb_position"] = (x["close"] - x["bb_lower"]) / (x["bb_upper"] - x["bb_lower"])
    
    # 신호 생성 (수정된 방식)
    # 1단계: EMA 크로스 + 가격변화
    stage1_long = (x["cr_prev"] <= 0) & (x["cr"] > 0) & (x["price_change"] > price_change_threshold)
    stage1_short = (x["cr_prev"] >= 0) & (x["cr"] < 0) & (x["price_change"] < -price_change_threshold)
    
    # 2단계: RSI + BB 조건 (확인 바 수만큼 앞으로 확인)
    stage2_long = pd.Series(False, index=x.index)
    stage2_short = pd.Series(False, index=x.index)
    
    for i in range(confirmation_bars, len(x)):
        # LONG: RSI 과매도 + BB 하단 근처
        if stage1_long.iloc[i]:
            # 확인 바 수만큼 앞으로 RSI와 BB 조건 확인
            rsi_condition = x["rsi"].iloc[i:i+confirmation_bars].min() < rsi_oversold
            bb_condition = x["bb_position"].iloc[i:i+confirmation_bars].min() < 0.4  # 더 관대하게
            stage2_long.iloc[i] = rsi_condition and bb_condition
        
        # SHORT: RSI 과매수 + BB 상단 근처
        if stage1_short.iloc[i]:
            # 확인 바 수만큼 앞으로 RSI와 BB 조건 확인
            rsi_condition = x["rsi"].iloc[i:i+confirmation_bars].max() > rsi_overbought
            bb_condition = x["bb_position"].iloc[i:i+confirmation_bars].max() > 0.6  # 더 관대하게
            stage2_short.iloc[i] = rsi_condition and bb_condition
    
    # 최종 신호
    sig = np.where(stage2_long, 1, np.where(stage2_short, -1, 0))
    x["sig"] = sig
    
    return x[["open", "high", "low", "close", "volume", f"ema{ema_fast}", f"ema{ema_slow}", 
              "price_change", "rsi", "bb_upper", "bb_lower", "bb_position", "sig"]].dropna()

def apply_trailing_stage2_fixed(pos: Position, price_now: float):
    """2단계 (수정): 트레일링 스탑"""
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

def stop_hit_stage2_fixed(pos: Position, price_now: float):
    """2단계 (수정): 손절"""
    if pos.side == 1:
        return price_now <= pos.entry * (1 - SL_PCT)
    else:
        return price_now >= pos.entry * (1 + SL_PCT)

def backtest_stage2_fixed(df: pd.DataFrame, **params):
    """2단계 (수정): 백테스트"""
    try:
        x = signal_engine_stage2_fixed(df, **params)
        
        if len(x) < 100:  # 데이터 부족
            return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100, 'win_rate': 0}
        
        eq = 10000.0
        equity = []
        trades = []
        pos = None
        
        # 빠른 처리를 위해 5분마다 샘플링
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
                if stop_hit_stage2_fixed(pos, px) or apply_trailing_stage2_fixed(pos, px):
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
            return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100, 'win_rate': 0}
        
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
        return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100, 'win_rate': 0}

def optimize_stage2_fixed():
    """🚀 2단계 (수정): RSI + 볼린저밴드 파라미터 최적화"""
    print("=== 🚀 2단계 최적화 시작 (수정): RSI + 볼린저밴드 ===")
    print(f"1단계 최적 파라미터 (고정): {STAGE1_BEST_PARAMS}")
    
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
    
    # 🚀 2단계: 최근 6개월 사용 (약 260,000행)
    recent_data = df.iloc[-int(len(df) * 0.05):]  # 최근 5%만 사용
    train_size = int(len(recent_data) * 0.7)
    train_data = recent_data.iloc[:train_size]
    test_data = recent_data.iloc[train_size:]
    
    print(f"전체 데이터: {len(df):,}행")
    print(f"🚀 2단계 최적화용 데이터: {len(recent_data):,}행 (최근 6개월)")
    print(f"훈련 데이터: {len(train_data):,}행")
    print(f"검증 데이터: {len(test_data):,}행")
    
    # 파라미터 조합 생성
    param_combinations = list(product(
        PARAM_RANGES_STAGE2['rsi_period'],
        PARAM_RANGES_STAGE2['rsi_oversold'],
        PARAM_RANGES_STAGE2['rsi_overbought'],
        PARAM_RANGES_STAGE2['bb_period'],
        PARAM_RANGES_STAGE2['bb_std'],
        PARAM_RANGES_STAGE2['confirmation_bars']
    ))
    
    # RSI 조건 필터링 (oversold < overbought)
    valid_combinations = [
        (rsi_p, rsi_ovs, rsi_ovb, bb_p, bb_s, conf_bars) 
        for rsi_p, rsi_ovs, rsi_ovb, bb_p, bb_s, conf_bars in param_combinations 
        if rsi_ovs < rsi_ovb
    ]
    
    print(f"🚀 2단계 테스트할 파라미터 조합: {len(valid_combinations)}개")
    print(f"🚀 2단계 예상 실행 시간: 약 {len(valid_combinations) * 0.4 / 60:.1f}분")
    
    # 최적화 실행
    best_score = -999
    best_params = None
    results = []
    
    for i, (rsi_p, rsi_ovs, rsi_ovb, bb_p, bb_s, conf_bars) in enumerate(valid_combinations):
        if i % 50 == 0:  # 진행률 표시
            print(f"진행률: {i+1}/{len(valid_combinations)} ({(i+1)/len(valid_combinations)*100:.1f}%)")
        
        params = {
            'rsi_period': rsi_p,
            'rsi_oversold': rsi_ovs,
            'rsi_overbought': rsi_ovb,
            'bb_period': bb_p,
            'bb_std': bb_s,
            'confirmation_bars': conf_bars
        }
        
        # 훈련 데이터로 백테스트
        result = backtest_stage2_fixed(train_data, **params)
        result['params'] = params
        results.append(result)
        
        # 최고 점수 업데이트
        if result['score'] > best_score:
            best_score = result['score']
            best_params = params
            print(f"🔥 새로운 최고점수: {best_score:.2f}")
            print(f"   파라미터: RSI={rsi_p}기간/{rsi_ovs}과매도/{rsi_ovb}과매수, BB={bb_p}기간/{bb_s}표준편차, 확인={conf_bars}바")
            print(f"   수익률: {result['return']:+.2f}%, 승률: {result['win_rate']:.1f}%")
    
    # 결과 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    top_20 = results[:20]  # TOP 20
    
    print("\n" + "="*80)
    print("🏆 2단계 최적화 결과 (수정) - TOP 20")
    print("="*80)
    
    for i, result in enumerate(top_20):
        params = result['params']
        print(f"{i+1:2d}. Score: {result['score']:7.2f} | "
              f"수익률: {result['return']:+6.2f}% | "
              f"Sharpe: {result['sharpe']:6.2f} | "
              f"승률: {result['win_rate']:5.1f}% | "
              f"거래: {result['trades']:3d}건")
        print(f"     RSI: {params['rsi_period']:2d}기간/{params['rsi_oversold']:2d}과매도/{params['rsi_overbought']:2d}과매수, "
              f"BB: {params['bb_period']:2d}기간/{params['bb_std']:3.1f}표준편차, 확인:{params['confirmation_bars']:1d}바")
        print()
    
    # 최적 파라미터로 검증 데이터 테스트
    print("="*80)
    print("🔍 2단계 최적 파라미터 검증 (Out-of-Sample)")
    print("="*80)
    
    if best_params is not None:
        validation_result = backtest_stage2_fixed(test_data, **best_params)
        print(f"2단계 최적 파라미터: {best_params}")
        print(f"검증 결과:")
        print(f"  수익률: {validation_result['return']:+.2f}%")
        print(f"  Sharpe: {validation_result['sharpe']:.2f}")
        print(f"  승률: {validation_result['win_rate']:.1f}%")
        print(f"  Max DD: {validation_result['max_dd']:+.1f}%")
        print(f"  거래 수: {validation_result['trades']}건")
        
        # 최종 결과 저장 (1단계 + 2단계 통합)
        final_params = {**STAGE1_BEST_PARAMS, **best_params}
        
        stage2_fixed_result = {
            'stage1_params': STAGE1_BEST_PARAMS,
            'stage2_params': best_params,
            'final_params': final_params,
            'validation_result': validation_result,
            'top_20': [{'params': r['params'], 'score': r['score'], 'return': r['return'], 'sharpe': r['sharpe'], 'win_rate': r['win_rate'], 'trades': r['trades']} for r in top_20]
        }
        
        with open('stage2_fixed_result.json', 'w') as f:
            json.dump(stage2_fixed_result, f, indent=2)
        
        print(f"\n✅ 2단계 결과 (수정)가 'stage2_fixed_result.json'에 저장되었습니다")
        print(f"🚀 최종 통합 파라미터: {final_params}")
        print(f"🚀 다음 단계: 3단계 리스크 관리 최적화")
        
    else:
        print("❌ 최적 파라미터를 찾지 못했습니다")
    
    return best_params, validation_result if best_params is not None else None

if __name__ == "__main__":
    optimize_stage2_fixed()
