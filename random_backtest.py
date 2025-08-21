"""
랜덤 백테스트 - 기존 학습된 모델로 3개 랜덤 기간 테스트
다양한 시장 상황에서의 모델 성과 평가
"""

import numpy as np
import pandas as pd
import random
import os
import torch
import pickle
from datetime import datetime, timedelta
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset
from src.models.nhits_model import Predictor

def random_backtest():
    """기존 모델로 랜덤 백테스트 실행"""
    
    # 설정 로드
    cfg = load_config()
    log = setup_logger(level="INFO")
    
    log.info("=== 기존 모델 랜덤 백테스트 시작 ===")
    
    # 데이터 로드 (전체 10년)
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    raw = db.load(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"])
    
    log.info(f"전체 데이터: {len(raw):,}행")
    log.info(f"데이터 기간: {raw['open_time'].min()} ~ {raw['open_time'].max()}")
    
    # 기존 모델 파일 확인
    model_path = os.path.join(os.path.dirname(__file__), "models", "predictor.pkl")
    if not os.path.exists(model_path):
        log.error(f"모델 파일이 없습니다: {model_path}")
        log.info("먼저 train.py를 실행해서 모델을 학습하세요.")
        return
    
    # 3개월 데이터 크기 (분 단위)
    three_months_size = 3 * 30 * 24 * 60  # 3개월
    
    # 랜덤 시드 설정 (재현 가능하도록)
    random.seed(42)
    
    # 3개 기간 랜덤 선택 (학습 데이터와 겹치지 않게)
    test_periods = []
    total_periods = len(raw) - three_months_size
    
    # 학습 데이터는 보통 마지막 부분이므로, 앞쪽에서 랜덤 선택
    for i in range(3):
        # 앞쪽 70% 구간에서 선택 (학습 데이터와 겹치지 않게)
        max_start = int(total_periods * 0.7)
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + three_months_size
        
        period_data = raw.iloc[start_idx:end_idx].copy()
        start_time = period_data['open_time'].iloc[0]
        end_time = period_data['open_time'].iloc[-1]
        
        test_periods.append({
            'period': i + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': start_time,
            'end_time': end_time,
            'data': period_data
        })
        
        log.info(f"기간 {i+1}: {start_time} ~ {end_time} ({len(period_data):,}행)")
    
    # 기존 모델 로드
    log.info("기존 모델 로드 중...")
    
    # 특징 생성 (전체 데이터로)
    log.info("전체 데이터로 특징 생성 중...")
    feat = enrich(raw)
    
    # 모델 초기화 (특징 수는 전체 데이터 기준)
    model = Predictor(
        in_feat=feat.shape[1],
        horizons=(60, 240, 1440),
        quantiles=(0.1, 0.5, 0.9),
        lr=9e-5
    )
    
    # 모델 로드 (pickle 방식)
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        log.info("모델 로드 완료")
    except Exception as e:
        log.error(f"모델 로드 실패: {e}")
        return
    
    # 각 기간별 백테스트 실행
    results = []
    
    for period_info in test_periods:
        log.info(f"\n--- 기간 {period_info['period']} 백테스트 시작 ---")
        
        # 해당 기간의 특징 데이터
        period_feat = feat.iloc[period_info['start_idx']:period_info['end_idx']]
        
        # 데이터셋 생성
        X, Y = make_dataset(period_feat, 512, (60, 240, 1440))
        
        if len(X) < 1000:
            log.warning(f"기간 {period_info['period']}: 데이터 부족 ({len(X)} < 1000)")
            continue
        
        # Train/Valid 분할 (80:20)
        split_idx = int(len(X) * 0.8)
        X_train, Y_train = X[:split_idx], Y[:split_idx]
        X_valid, Y_valid = X[split_idx:], Y[split_idx:]
        
        # 예측 (모델 학습 없이)
        log.info("예측 중...")
        predictions = []
        chunk_size = 500
        
        for i in range(0, len(X_valid), chunk_size):
            end = min(i + chunk_size, len(X_valid))
            batch_pred = model.predict(X_valid[i:end])
            for pred in batch_pred:
                predictions.append(pred[0])  # 1시간 예측값
        
        # 가격 데이터
        valid_start = 512 + split_idx
        valid_prices = period_feat["close"].iloc[valid_start:valid_start+len(predictions)].values
        
        log.info(f"예측 완료: {len(predictions)}개")
        log.info(f"가격 데이터: {len(valid_prices)}개")
        
        # 거래 시뮬레이션
        log.info("거래 시뮬레이션 시작...")
        trades = simulate_trading_detailed(valid_prices, predictions, 0.2)
        
        # 성과 분석
        performance = analyze_performance(trades, valid_prices)
        performance['period'] = period_info['period']
        performance['start_time'] = period_info['start_time']
        performance['end_time'] = period_info['end_time']
        performance['data_points'] = len(valid_prices)
        
        results.append(performance)
        
        log.info(f"기간 {period_info['period']} 완료: {performance['total_return']:+.2f}%")
    
    # 전체 결과 종합
    if results:
        summarize_results(results)
    else:
        log.error("모든 기간에서 백테스트 실패")

def simulate_trading_detailed(prices, predictions, confidence_threshold):
    """상세한 거래 시뮬레이션 - Trailing Stop 적용"""
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
                    'time': i
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
                    exit_reason = "trailing_stop"
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
                    'time': i
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
        
        trades.append({
            'type': 'exit',
            'price': final_price,
            'pnl_pct': final_pnl,
            'pnl_amount': pnl_amount,
            'peak_pnl': peak_pnl,
            'reason': 'force_close',
            'time': len(prices) - 1
        })
    
    return trades

def analyze_performance(trades, prices):
    """거래 성과 분석"""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_time': 0.0
        }
    
    # 거래별 수익률 계산
    returns = []
    trade_pairs = []
    
    for i in range(len(trades)):
        if trades[i]['type'] == 'entry':
            for j in range(i + 1, len(trades)):
                if trades[j]['type'] == 'exit':
                    trade_pairs.append((trades[i], trades[j]))
                    returns.append(trades[j]['pnl_pct'])
                    break
    
    if len(returns) == 0:
        return {
            'total_trades': len(trades),
            'total_return': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_time': 0.0
        }
    
    # 기본 메트릭
    total_return = np.prod([1 + r for r in returns]) - 1
    win_rate = len([r for r in returns if r > 0]) / len(returns)
    
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    max_win = max(returns) if returns else 0.0
    max_loss = min(returns) if returns else 0.0
    
    # Sharpe Ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 0.01
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    
    # Profit Factor
    total_wins = sum(wins) if wins else 0.0
    total_losses = sum(abs(l) for l in losses) if losses else 0.01
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
    
    # Max Drawdown
    cumulative = np.cumprod([1 + r for r in returns])
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # 평균 보유 시간
    holding_times = [t.get('holding_time', 0) for t in trades if t['type'] == 'exit']
    avg_holding_time = np.mean(holding_times) if holding_times else 0.0
    
    return {
        'total_trades': len(trade_pairs),
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_holding_time': avg_holding_time
    }

def summarize_results(results):
    """전체 결과 종합"""
    print("\n" + "="*80)
    print("🎯 기존 모델 랜덤 백테스트 결과 종합")
    print("="*80)
    
    # 각 기간별 결과
    for result in results:
        print(f"\n📊 기간 {result['period']}: {result['start_time']} ~ {result['end_time']}")
        print(f"   거래 수: {result['total_trades']:2d}건")
        print(f"   총 수익률: {result['total_return']*100:+.2f}%")
        print(f"   승률: {result['win_rate']*100:5.1f}%")
        print(f"   Sharpe: {result['sharpe_ratio']:6.2f}")
        print(f"   Profit Factor: {result['profit_factor']:5.2f}")
        print(f"   Max Drawdown: {result['max_drawdown']*100:+.1f}%")
    
    # 전체 평균
    if results:
        avg_return = np.mean([r['total_return'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        total_trades = sum([r['total_trades'] for r in results])
        
        print(f"\n🏆 전체 평균 성과:")
        print(f"   평균 수익률: {avg_return*100:+.2f}%")
        print(f"   평균 승률: {avg_win_rate*100:5.1f}%")
        print(f"   평균 Sharpe: {avg_sharpe:6.2f}")
        print(f"   평균 Profit Factor: {avg_profit_factor:5.2f}")
        print(f"   평균 Max Drawdown: {avg_drawdown*100:+.1f}%")
        print(f"   총 거래 수: {total_trades}건")
        
        # 안정성 평가
        returns = [r['total_return'] for r in results]
        std_return = np.std(returns)
        consistency = 1 - std_return / abs(avg_return) if abs(avg_return) > 0 else 0
        
        print(f"\n📈 안정성 지표:")
        print(f"   수익률 표준편차: {std_return*100:.2f}%")
        print(f"   일관성 점수: {consistency:.2f}")
        
        if consistency > 0.7:
            print("   ✅ 높은 일관성: 모델이 안정적")
        elif consistency > 0.4:
            print("   ⚠️  보통 일관성: 모델이 어느 정도 안정적")
        else:
            print("   ❌ 낮은 일관성: 모델이 불안정")
    
    print("="*80)

if __name__ == "__main__":
    random_backtest()
