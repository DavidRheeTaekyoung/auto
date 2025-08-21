# -*- coding: utf-8 -*-
"""
Simple Reversal 전략 최적화
10년 데이터를 사용해서 최적 파라미터 찾기
"""

import pandas as pd
import numpy as np
import random
from itertools import product
from dataclasses import dataclass
from src.utils.config import load_config
from src.processors.data_storage import DataStorage

# =========================
# 최적화 파라미터 범위 (전체 최적화)
# =========================
PARAM_RANGES = {
    # 진입 조건 관련
    'price_change_period': [1, 3, 5, 10, 15],  # 가격 변화 측정 기간 (분)
    'price_change_threshold': [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005],  # 가격 변화 임계값 (0.01% ~ 0.5%)
    
    # EMA 관련
    'ema_fast': [3, 5, 7, 9, 12, 15],  # 빠른 EMA
    'ema_slow': [12, 15, 18, 21, 26, 30],  # 느린 EMA
    
    # 추가 진입 조건
    'volume_threshold': [1.0, 1.5, 2.0, 2.5, 3.0],  # 거래량 배수 (평균 대비)
    'rsi_period': [7, 14, 21],  # RSI 기간
    'rsi_oversold': [20, 25, 30],  # RSI 과매도
    'rsi_overbought': [70, 75, 80],  # RSI 과매수
    
    # 추가 기술적 지표
    'bb_period': [10, 20, 30],  # 볼린저 밴드 기간
    'bb_std': [1.5, 2.0, 2.5],  # 볼린저 밴드 표준편차
    
    # 진입 지연
    'entry_delay': [0, 1, 2, 3],  # 신호 발생 후 진입 지연 (분)
    
    # 진입 확인
    'confirmation_bars': [1, 2, 3, 5],  # 진입 확인을 위한 추가 바 수
    
    # 리스크 관리 (최적화 대상)
    'leverage': [3, 5, 8, 10, 15],  # 레버리지
    'stop_loss_pct': [0.001, 0.002, 0.003, 0.005, 0.01],  # 손절 비율 (0.1% ~ 1%)
    'take_profit_arm_pct': [0.003, 0.005, 0.01, 0.015, 0.02],  # 트레일링 활성화 (0.3% ~ 2%)
    'giveback_pct': [0.1, 0.15, 0.2, 0.25, 0.3],  # 이익 반납 (10% ~ 30%)
    
    # 거래량 관련
    'min_volume': [1000, 5000, 10000, 50000],  # 최소 거래량 (USD)
}

# 고정 파라미터
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

def signal_engine_optimized(df: pd.DataFrame, **params):
    """확장된 최적화 가능한 신호 엔진"""
    x = df.copy()
    
    # 기본 파라미터 추출
    ema_fast = params.get('ema_fast', 9)
    ema_slow = params.get('ema_slow', 21)
    price_change_period = params.get('price_change_period', 10)
    price_change_threshold = params.get('price_change_threshold', 0.002)
    volume_threshold = params.get('volume_threshold', 1.5)
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    entry_delay = params.get('entry_delay', 0)
    confirmation_bars = params.get('confirmation_bars', 1)
    
    # EMA 계산
    x[f"ema{ema_fast}"] = ema(x["close"], ema_fast)
    x[f"ema{ema_slow}"] = ema(x["close"], ema_slow)
    x["cr"] = np.sign(x[f"ema{ema_fast}"] - x[f"ema{ema_slow}"])
    x["cr_prev"] = x["cr"].shift(1)
    
    # 가격 변화율
    x["price_change"] = x["close"].pct_change(price_change_period)
    
    # RSI
    delta = x["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    x["rsi"] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    x["bb_middle"] = x["close"].rolling(window=bb_period).mean()
    x["bb_std"] = x["close"].rolling(window=bb_period).std()
    x["bb_upper"] = x["bb_middle"] + (x["bb_std"] * bb_std)
    x["bb_lower"] = x["bb_middle"] - (x["bb_std"] * bb_std)
    x["bb_position"] = (x["close"] - x["bb_lower"]) / (x["bb_upper"] - x["bb_lower"])
    
    # 거래량 조건
    x["volume_ma"] = x["volume"].rolling(window=20).mean()
    x["volume_ratio"] = x["volume"] / x["volume_ma"]
    
    # 기본 신호 (EMA 크로스 + 가격 변화)
    base_sig = np.where(
        (x["cr_prev"] <= 0) & (x["cr"] > 0) & (x["price_change"] > price_change_threshold), 1,
        np.where(
            (x["cr_prev"] >= 0) & (x["cr"] < 0) & (x["price_change"] < -price_change_threshold), -1, 0
        )
    )
    
    # 추가 필터 적용
    long_conditions = (
        (base_sig == 1) &
        (x["rsi"] < rsi_overbought) &  # RSI 과매수 아님
        (x["volume_ratio"] > volume_threshold) &  # 거래량 증가
        (x["bb_position"] < 0.8)  # 볼린저 밴드 상단 근처 아님
    )
    
    short_conditions = (
        (base_sig == -1) &
        (x["rsi"] > rsi_oversold) &  # RSI 과매도 아님
        (x["volume_ratio"] > volume_threshold) &  # 거래량 증가
        (x["bb_position"] > 0.2)  # 볼린저 밴드 하단 근처 아님
    )
    
    # 최종 신호
    x["sig"] = np.where(long_conditions, 1, np.where(short_conditions, -1, 0))
    
    # 진입 지연 및 확인
    if entry_delay > 0:
        x["sig"] = x["sig"].shift(entry_delay)
    
    if confirmation_bars > 1:
        # 연속 N개 바에서 같은 신호인지 확인
        for i in range(1, confirmation_bars):
            x["sig"] = np.where(
                (x["sig"] == x["sig"].shift(i)) & (x["sig"] != 0),
                x["sig"], 0
            )
    
    return x[["open", "high", "low", "close", "volume", f"ema{ema_fast}", f"ema{ema_slow}", 
              "price_change", "rsi", "bb_position", "volume_ratio", "sig"]].dropna()

def apply_trailing_optimized(pos: Position, price_now: float, arm_pct: float, giveback: float):
    """최적화된 트레일링 스탑"""
    if pos.side == 1:
        gain = (price_now / pos.entry) - 1.0
        if not pos.armed and gain >= arm_pct:
            pos.armed = True
            pos.peak = max(pos.peak, price_now)
        if pos.armed:
            pos.peak = max(pos.peak, price_now)
            trigger = pos.entry * (1 + (pos.peak / pos.entry - 1) * (1 - giveback))
            if price_now <= trigger:
                return True
        return False
    else:
        gain = (pos.entry / price_now) - 1.0
        if not pos.armed and gain >= arm_pct:
            pos.armed = True
            pos.peak = min(pos.peak, price_now)
        if pos.armed:
            pos.peak = min(pos.peak, price_now)
            trigger = pos.entry * (1 - (1 - pos.peak / pos.entry) * (1 - giveback))
            if price_now >= trigger:
                return True
        return False

def backtest_optimized(df: pd.DataFrame, **params):
    """확장된 최적화용 백테스트"""
    try:
        x = signal_engine_optimized(df, **params)
        
        if len(x) < 100:  # 데이터 부족
            return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100}
        
        # 파라미터에서 리스크 관리 값 추출
        leverage = params.get('leverage', 5)
        stop_loss_pct = params.get('stop_loss_pct', 0.003)
        take_profit_arm_pct = params.get('take_profit_arm_pct', 0.01)
        giveback_pct = params.get('giveback_pct', 0.2)
        min_volume = params.get('min_volume', 10000)
        
        eq = 10000.0
        equity = []
        trades = []
        pos = None
        
        for t, row in x.iterrows():
            px = float(row["close"])
            sig = int(row["sig"])
            volume_usd = float(row["volume"]) * px  # 거래량을 USD로 변환
            
            # 최소 거래량 조건 확인
            if volume_usd < min_volume:
                continue
            
            # 진입
            if pos is None and sig != 0:
                qty = (NOTIONAL_USD * leverage) / px
                pos = Position(side=sig, entry=px, qty=qty, peak=px, armed=False)
                eq -= (qty * px) * (FEE_BP / 2 / 1e4 + SLIP_BP / 2 / 1e4)
            
            # 청산
            if pos is not None:
                # 손절 확인
                if pos.side == 1 and px <= pos.entry * (1 - stop_loss_pct):
                    pnl = pos.qty * (px - pos.entry) * pos.side
                    cost = (pos.qty * px) * (FEE_BP / 2 / 1e4 + SLIP_BP / 2 / 1e4)
                    eq += pnl - cost
                    trades.append({
                        "time": t, 
                        "side": "LONG", "entry": pos.entry, "exit": px, 
                        "pnl": pnl, "eq": eq, "reason": "STOP_LOSS"
                    })
                    pos = None
                elif pos.side == -1 and px >= pos.entry * (1 + stop_loss_pct):
                    pnl = pos.qty * (px - pos.entry) * pos.side
                    cost = (pos.qty * px) * (FEE_BP / 2 / 1e4 + SLIP_BP / 2 / 1e4)
                    eq += pnl - cost
                    trades.append({
                        "time": t, 
                        "side": "SHORT", "entry": pos.entry, "exit": px, 
                        "pnl": pnl, "eq": eq, "reason": "STOP_LOSS"
                    })
                    pos = None
                # 트레일링 스탑 확인
                elif apply_trailing_optimized(pos, px, take_profit_arm_pct, giveback_pct):
                    pnl = pos.qty * (px - pos.entry) * pos.side
                    cost = (pos.qty * px) * (FEE_BP / 2 / 1e4 + SLIP_BP / 2 / 1e4)
                    eq += pnl - cost
                    trades.append({
                        "time": t, 
                        "side": "LONG" if pos.side == 1 else "SHORT",
                        "entry": pos.entry, "exit": px, 
                        "pnl": pnl, "eq": eq, "reason": "TRAILING"
                    })
                    pos = None
            
            equity.append(eq if pos is None else eq + pos.qty * (px - pos.entry) * pos.side)
        
        # 성과 계산
        if len(trades) == 0:
            return {'score': -999, 'trades': 0, 'return': -100, 'sharpe': -999, 'max_dd': -100}
        
        final_return = (eq - 10000) / 10000
        
        # Sharpe 계산
        eq_series = pd.Series(equity)
        returns = eq_series.pct_change().fillna(0)
        sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(365 * 24 * 60)
        
        # Max Drawdown
        dd = (eq_series / eq_series.cummax() - 1).min()
        
        # 승률
        trade_returns = [(t['pnl'] / (t['entry'] * leverage / 10000)) for t in trades]
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

def optimize_parameters():
    """파라미터 최적화 실행"""
    print("=== Simple Reversal 전략 최적화 시작 ===")
    
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
    
    # 최적화용 데이터 (전체 70% 사용, 30%는 검증용)
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    print(f"훈련 데이터: {len(train_data):,}행")
    print(f"검증 데이터: {len(test_data):,}행")
    
    # 파라미터 조합 생성 (전체 최적화)
    param_combinations = list(product(
        PARAM_RANGES['price_change_period'],
        PARAM_RANGES['price_change_threshold'], 
        PARAM_RANGES['ema_fast'],
        PARAM_RANGES['ema_slow'],
        PARAM_RANGES['volume_threshold'],
        PARAM_RANGES['rsi_period'],
        PARAM_RANGES['rsi_oversold'],
        PARAM_RANGES['rsi_overbought'],
        PARAM_RANGES['bb_period'],
        PARAM_RANGES['bb_std'],
        PARAM_RANGES['entry_delay'],
        PARAM_RANGES['confirmation_bars'],
        PARAM_RANGES['leverage'],
        PARAM_RANGES['stop_loss_pct'],
        PARAM_RANGES['take_profit_arm_pct'],
        PARAM_RANGES['giveback_pct'],
        PARAM_RANGES['min_volume']
    ))
    
    # EMA 조건 필터링 (fast < slow)
    valid_combinations = []
    for combo in param_combinations:
        (period, threshold, fast, slow, volume_thresh, rsi_per, rsi_ovs, rsi_ovb,
         bb_per, bb_std, delay, confirm, lev, sl, tp, gb, min_vol) = combo
        
        # 기본 필터링
        if fast >= slow:  # EMA 조건
            continue
        if rsi_ovs >= rsi_ovb:  # RSI 조건
            continue
        if sl >= tp:  # 손절이 익절보다 크면 안됨
            continue
            
        valid_combinations.append(combo)
    
    print(f"테스트할 파라미터 조합: {len(valid_combinations)}개")
    
    # 최적화 실행
    best_score = -999
    best_params = None
    results = []
    
    for i, combo in enumerate(valid_combinations):
        if i % 10 == 0:
            print(f"진행률: {i+1}/{len(valid_combinations)} ({(i+1)/len(valid_combinations)*100:.1f}%)")
        
        (period, threshold, fast, slow, volume_thresh, rsi_per, rsi_ovs, rsi_ovb,
         bb_per, bb_std, delay, confirm, lev, sl, tp, gb, min_vol) = combo
        
        params = {
            'price_change_period': period,
            'price_change_threshold': threshold,
            'ema_fast': fast,
            'ema_slow': slow,
            'volume_threshold': volume_thresh,
            'rsi_period': rsi_per,
            'rsi_oversold': rsi_ovs,
            'rsi_overbought': rsi_ovb,
            'bb_period': bb_per,
            'bb_std': bb_std,
            'entry_delay': delay,
            'confirmation_bars': confirm,
            'leverage': lev,
            'stop_loss_pct': sl,
            'take_profit_arm_pct': tp,
            'giveback_pct': gb,
            'min_volume': min_vol
        }
        
        # 훈련 데이터로 백테스트
        result = backtest_optimized(train_data, **params)
        result['params'] = params
        results.append(result)
        
        # 최고 점수 업데이트
        if result['score'] > best_score:
            best_score = result['score']
            best_params = params
            print(f"🔥 새로운 최고점수: {best_score:.2f}")
            print(f"   파라미터: {params}")
            print(f"   수익률: {result['return']:+.2f}%, 승률: {result['win_rate']:.1f}%")
    
    # 결과 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    top_10 = results[:10]
    
    print("\n" + "="*80)
    print("🏆 최적화 결과 - TOP 10")
    print("="*80)
    
    for i, result in enumerate(top_10):
        params = result['params']
        print(f"{i+1:2d}. Score: {result['score']:7.2f} | "
              f"수익률: {result['return']:+6.2f}% | "
              f"Sharpe: {result['sharpe']:6.2f} | "
              f"승률: {result['win_rate']:5.1f}% | "
              f"거래: {result['trades']:3d}건")
        print(f"     핵심 파라미터:")
        print(f"       EMA: {params['ema_fast']}/{params['ema_slow']}, "
              f"가격변화: {params['price_change_period']}분/{params['price_change_threshold']:.3f}")
        print(f"       레버리지: {params['leverage']}x, "
              f"손절: {params['stop_loss_pct']:.3f}, "
              f"익절: {params['take_profit_arm_pct']:.3f}")
        print(f"       RSI: {params['rsi_period']}기간, "
              f"과매도: {params['rsi_oversold']}, "
              f"과매수: {params['rsi_overbought']}")
        print()
    
    # 최적 파라미터로 검증 데이터 테스트
    print("="*80)
    print("🔍 최적 파라미터 검증 (Out-of-Sample)")
    print("="*80)
    
    if best_params is not None:
        validation_result = backtest_optimized(test_data, **best_params)
        print(f"최적 파라미터: {best_params}")
        print(f"검증 결과:")
        print(f"  수익률: {validation_result['return']:+.2f}%")
        print(f"  Sharpe: {validation_result['sharpe']:.2f}")
        print(f"  승률: {validation_result['win_rate']:.1f}%")
        print(f"  Max DD: {validation_result['max_dd']:+.1f}%")
        print(f"  거래 수: {validation_result['trades']}건")
    else:
        print("❌ 최적 파라미터를 찾지 못했습니다")
        print("모든 파라미터 조합이 매우 나쁜 결과를 보였습니다")
    
    # 최적 파라미터 저장
    import json
    with open('best_reversal_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✅ 최적 파라미터가 'best_reversal_params.json'에 저장되었습니다")
    
    return best_params, validation_result

if __name__ == "__main__":
    optimize_parameters()
