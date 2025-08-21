"""
거래 시뮬레이션 테스트 - 실제 손실 금액 확인
"""

import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset
from src.models.nhits_model import Predictor

def test_trading_performance():
    """거래 성과 테스트"""
    
    # 설정 로드
    cfg = load_config()
    log = setup_logger(level="DEBUG")
    
    log.info("=== 거래 성과 테스트 시작 ===")
    
    # 데이터 로드 (최근 1개월)
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    raw = db.load(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"])
    
    # 최근 1개월 데이터
    test_data = raw.tail(30*24*60)
    log.info(f"테스트 데이터: {len(test_data):,}행")
    
    # 특징 생성
    feat = enrich(test_data)
    X, Y = make_dataset(feat, 512, (60, 240, 1440))
    
    if len(X) < 100:
        log.error("데이터 부족")
        return
    
    # Train/Valid 분할
    split_idx = int(len(X) * 0.8)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_valid, Y_valid = X[split_idx:], Y[split_idx:]
    
    # 모델 학습 (빠르게)
    model = Predictor(
        in_feat=X.shape[2],
        horizons=(60, 240, 1440),
        quantiles=(0.1, 0.5, 0.9),
        lr=9e-5  # 최적화된 파라미터
    )
    
    log.info("모델 학습 중...")
    model.fit(X_train, Y_train, epochs=20, batch=128)
    
    # 예측
    log.info("예측 중...")
    predictions = []
    for i in range(0, len(X_valid), 500):
        end = min(i + 500, len(X_valid))
        batch_pred = model.predict(X_valid[i:end])
        for pred in batch_pred:
            predictions.append(pred[0])  # 1시간 예측값
    
    # 가격 데이터
    valid_start = 512 + split_idx
    valid_prices = feat["close"].iloc[valid_start:valid_start+len(predictions)].values
    
    log.info(f"예측 완료: {len(predictions)}개")
    log.info(f"가격 데이터: {len(valid_prices)}개")
    
    # 거래 시뮬레이션
    log.info("거래 시뮬레이션 시작...")
    trades = simulate_trading_detailed(valid_prices, predictions, 0.2)
    
    # 성과 분석
    analyze_trades_detailed(trades, valid_prices)

def simulate_trading_detailed(prices, predictions, confidence_threshold):
    """상세한 거래 시뮬레이션"""
    trades = []
    cash = 10000
    pos_qty = 0.0
    entry = None
    side = None
    entry_idx = None
    
    # Trailing Stop 변수
    peak_price = None
    peak_pnl = None
    trailing_activated = False
    
    for i, (price, pred) in enumerate(zip(prices, predictions)):
        # 예측값 처리
        if isinstance(pred, (list, tuple, np.ndarray)) and len(pred) >= 3:
            q10, q50, q90 = float(pred[0]), float(pred[1]), float(pred[2])
        else:
            q50 = float(pred)
            q10 = q50 * 0.99
            q90 = q50 * 1.01
        
        # 신뢰도 계산
        pred_range = abs(q90 - q10)
        confidence = 1.0 / (1.0 + pred_range / price * 100)
        
        # 예측 방향
        expected_return = (q50 - price) / price
        direction = 1 if expected_return > 0 else -1
        
        # 포지션 진입
        if pos_qty == 0:
            if confidence > confidence_threshold and abs(expected_return) > 0.001:
                pos_qty = (cash * 2) / price
                side = direction
                entry = price
                entry_idx = i
                
                peak_price = price
                peak_pnl = 0
                trailing_activated = False
                
                trades.append({
                    'type': 'entry',
                    'price': price,
                    'side': side,
                    'confidence': confidence,
                    'expected_return': expected_return,
                    'time': i,
                    'cash_before': cash
                })
        
        # 포지션 관리
        elif pos_qty != 0:
            holding_time = i - entry_idx
            current_pnl = (price - entry) / entry * side
            
            # 최고 수익 업데이트
            if current_pnl > peak_pnl:
                peak_pnl = current_pnl
                peak_price = price
                if peak_pnl >= 0.01:
                    trailing_activated = True
            
            # 청산 조건
            should_exit = False
            exit_reason = ""
            
            if current_pnl <= -0.02:
                should_exit = True
                exit_reason = "stop_loss"
            elif trailing_activated:
                drawdown_from_peak = (peak_pnl - current_pnl) / peak_pnl if peak_pnl > 0 else 0
                if drawdown_from_peak >= 0.3:
                    should_exit = True
                    exit_reason = f"trailing_stop"
                elif peak_pnl >= 0.01 and current_pnl < peak_pnl * 0.7:
                    should_exit = True
                    exit_reason = "profit_protection"
            elif holding_time >= 120:
                should_exit = True
                exit_reason = "timeout"
            elif current_pnl >= 0.05:
                should_exit = True
                exit_reason = "target_reached"
            
            if should_exit:
                # 실제 손익 계산
                pnl_amount = pos_qty * (price - entry) * side
                cash += pnl_amount
                
                trades.append({
                    'type': 'exit',
                    'price': price,
                    'pnl_pct': current_pnl,
                    'pnl_amount': pnl_amount,
                    'peak_pnl': peak_pnl,
                    'reason': exit_reason,
                    'holding_time': holding_time,
                    'time': i,
                    'cash_after': cash
                })
                
                # 포지션 초기화
                pos_qty = 0
                entry = None
                side = None
                entry_idx = None
                peak_price = None
                peak_pnl = None
                trailing_activated = False
    
    # 마지막 포지션 강제 청산
    if pos_qty != 0 and len(prices) > 0:
        final_price = prices[-1]
        final_pnl = (final_price - entry) / entry * side
        pnl_amount = pos_qty * (final_price - entry) * side
        cash += pnl_amount
        
        trades.append({
            'type': 'exit',
            'price': final_price,
            'pnl_pct': final_pnl,
            'pnl_amount': pnl_amount,
            'peak_pnl': peak_pnl,
            'reason': 'force_close',
            'time': len(prices) - 1,
            'cash_after': cash
        })
    
    return trades

def analyze_trades_detailed(trades, prices):
    """상세한 거래 분석"""
    if len(trades) == 0:
        print("거래가 없습니다.")
        return
    
    print("\n" + "="*60)
    print("거래 상세 분석")
    print("="*60)
    
    # 거래별 상세 정보
    entry_trades = [t for t in trades if t['type'] == 'entry']
    exit_trades = [t for t in trades if t['type'] == 'exit']
    
    print(f"총 거래 신호: {len(trades)}")
    print(f"진입 거래: {len(entry_trades)}")
    print(f"청산 거래: {len(exit_trades)}")
    
    if len(exit_trades) == 0:
        print("⚠️  청산된 거래가 없습니다!")
        return
    
    # 손익 분석
    total_pnl = sum([t['pnl_amount'] for t in exit_trades])
    initial_cash = 10000
    final_cash = exit_trades[-1]['cash_after']
    
    print(f"\n💰 자본 변화:")
    print(f"  초기 자본: {initial_cash:,.0f}원")
    print(f"  최종 자본: {final_cash:,.0f}원")
    print(f"  총 손익: {total_pnl:+,.0f}원")
    print(f"  수익률: {(total_pnl/initial_cash)*100:+.2f}%")
    
    # 거래별 상세
    print(f"\n📊 거래별 상세:")
    for i, trade in enumerate(trades):
        if trade['type'] == 'entry':
            print(f"  {i+1:2d}. 진입: {trade['price']:,.0f}원 (신뢰도: {trade['confidence']:.3f})")
        else:
            pnl_color = "🔴" if trade['pnl_amount'] < 0 else "🟢"
            print(f"  {i+1:2d}. 청산: {trade['price']:,.0f}원 | {pnl_color} {trade['pnl_amount']:+,.0f}원 ({trade['pnl_pct']*100:+.2f}%) | {trade['reason']}")
    
    # 성과 지표
    pnl_amounts = [t['pnl_amount'] for t in exit_trades]
    pnl_pcts = [t['pnl_pct'] for t in exit_trades]
    
    wins = [p for p in pnl_amounts if p > 0]
    losses = [p for p in pnl_amounts if p < 0]
    
    print(f"\n📈 성과 지표:")
    print(f"  승률: {len(wins)}/{len(exit_trades)} ({len(wins)/len(exit_trades)*100:.1f}%)")
    print(f"  평균 수익: {np.mean(wins):+,.0f}원" if wins else "  평균 수익: 없음")
    print(f"  평균 손실: {np.mean(losses):+,.0f}원" if losses else "  평균 손실: 없음")
    print(f"  최대 수익: {max(pnl_amounts):+,.0f}원")
    print(f"  최대 손실: {min(pnl_amounts):+,.0f}원")
    
    print("="*60)

if __name__ == "__main__":
    test_trading_performance()
