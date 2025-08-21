"""
수정된 앙상블 모델 랜덤 백테스트
전체 10년으로 훈련된 앙상블 모델을 사용하여 3개 랜덤 기간에서 백테스트
"""

import numpy as np
import pandas as pd
import random
import os
import pickle
from datetime import datetime, timedelta
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset
from src.models.nhits_model import Predictor
from src.models.ensemble_model import EnsembleModel

def create_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 특징 생성"""
    try:
        features = pd.DataFrame(index=data.index)
        
        # 기본 가격 특징
        features['close'] = data['close']
        features['volume'] = data['volume']
        
        # 이동평균
        features['ma_5'] = data['close'].rolling(5).mean()
        features['ma_20'] = data['close'].rolling(20).mean()
        features['ma_50'] = data['close'].rolling(50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # 변동성
        features['volatility'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        
        # 추세
        features['trend'] = (data['close'] - data['close'].rolling(50).mean()) / data['close'].rolling(50).mean()
        
        # 거래량 특징
        features['volume_ma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma']
        
        # 추가 기술적 지표
        features['price_change'] = data['close'].pct_change()
        features['price_change_5'] = data['close'].pct_change(5)
        features['price_change_20'] = data['close'].pct_change(20)
        
        # 볼린저 밴드
        bb_middle = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # NaN 제거
        features = features.dropna()
        
        return features
        
    except Exception as e:
        print(f"기술적 특징 생성 실패: {e}")
        return pd.DataFrame()

def ensemble_backtest_fixed():
    """수정된 앙상블 모델 랜덤 백테스트"""
    
    # 설정 로드
    cfg = load_config()
    log = setup_logger(level="INFO")
    
    log.info("=== 수정된 앙상블 모델 랜덤 백테스트 시작 ===")
    
    # 전체 데이터 로드
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    symbol_db = cfg["data"]["symbol_internal"].replace("/", "")
    raw = db.load(symbol_db, cfg["data"]["timeframe"])
    
    log.info(f"전체 데이터: {len(raw):,}행")
    log.info(f"데이터 기간: {raw['open_time'].min()} ~ {raw['open_time'].max()}")
    
    # 훈련된 앙상블 모델 로드
    ensemble_path = os.path.join(os.path.dirname(__file__), "models", "ensemble_model.pkl")
    if not os.path.exists(ensemble_path):
        log.error(f"앙상블 모델 파일이 없습니다: {ensemble_path}")
        log.info("먼저 train_ensemble_model.py를 실행해서 앙상블 모델을 훈련하세요.")
        return
    
    log.info("훈련된 앙상블 모델 로드 중...")
    with open(ensemble_path, 'rb') as f:
        ensemble = pickle.load(f)
    log.info("앙상블 모델 로드 완료")
    log.info(f"앙상블 모델 정보: {ensemble.get_model_info()}")
    
    # 전체 데이터로 특징 생성
    log.info("전체 데이터로 특징 생성 중...")
    feat = enrich(raw)
    log.info(f"특징 생성 완료: {feat.shape}")
    
    # 3개월 데이터 크기
    three_months_size = 3 * 30 * 24 * 60  # 3개월
    
    # 랜덤 시드 설정
    random.seed(42)
    
    # 백테스트용 구간에서 3개 기간 랜덤 선택 (전체의 70% 이후)
    test_start = int(len(raw) * 0.7)  # 훈련 데이터 이후
    test_periods = []
    available_periods = len(raw) - test_start - three_months_size
    
    log.info(f"백테스트 가능 구간: {test_start:,} ~ {len(raw):,}")
    
    for i in range(3):
        start_idx = test_start + random.randint(0, available_periods)
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
    
    # 각 기간별 백테스트 실행
    results = []
    
    for period_info in test_periods:
        log.info(f"\n--- 기간 {period_info['period']} 앙상블 백테스트 시작 ---")
        
        # 해당 기간의 특징 데이터
        period_feat = feat.iloc[period_info['start_idx']:period_info['end_idx']]
        
        # 데이터셋 생성
        X, Y = make_dataset(period_feat, 512, (60, 240, 1440))
        
        if len(X) < 1000:
            log.warning(f"기간 {period_info['period']}: 데이터 부족 ({len(X)} < 1000)")
            continue
        
        # Train/Valid 분할 (80:20)
        split_idx = int(len(X) * 0.8)
        X_valid = X[split_idx:]
        
        # 앙상블 예측
        log.info("앙상블 예측 중...")
        predictions = []
        confidences = []
        chunk_size = 200  # 작은 청크로 처리
        
        for i in range(0, len(X_valid), chunk_size):
            end = min(i + chunk_size, len(X_valid))
            
            # NHiTS 특징
            nhits_batch = X_valid[i:end]
            
            # 해당 기간의 기술적 특징 생성
            batch_start = 512 + split_idx + i
            batch_end = batch_start + (end - i)
            
            # 인덱스 범위 확인
            if batch_end > len(period_feat):
                batch_end = len(period_feat)
                actual_size = batch_end - batch_start
                nhits_batch = nhits_batch[:actual_size]
            
            if batch_start >= len(period_feat) or len(nhits_batch) == 0:
                break
            
            batch_feat = period_feat.iloc[batch_start:batch_end]
            technical_batch = create_technical_features(batch_feat)
            
            # 현재 가격
            current_prices = batch_feat['close'].values
            
            # 데이터 길이 맞춤
            min_len = min(len(nhits_batch), len(technical_batch), len(current_prices))
            if min_len <= 0:
                continue
            
            nhits_batch = nhits_batch[:min_len]
            technical_batch = technical_batch.iloc[:min_len] if len(technical_batch) > 0 else pd.DataFrame()
            current_prices = current_prices[:min_len]
            
            # 앙상블 예측
            try:
                batch_preds, batch_confs = ensemble.ensemble_predict(
                    nhits_batch, technical_batch, current_prices
                )
                
                predictions.extend(batch_preds)
                confidences.extend(batch_confs)
                
            except Exception as e:
                log.warning(f"청크 {i//chunk_size + 1} 예측 실패: {e}")
                continue
        
        if len(predictions) == 0:
            log.warning(f"기간 {period_info['period']}: 예측 실패")
            continue
        
        # 가격 데이터
        valid_start = 512 + split_idx
        valid_end = valid_start + len(predictions)
        valid_prices = period_feat["close"].iloc[valid_start:valid_end].values
        
        log.info(f"앙상블 예측 완료: {len(predictions)}개")
        log.info(f"가격 데이터: {len(valid_prices)}개")
        log.info(f"평균 신뢰도: {np.mean(confidences):.3f}")
        
        # 거래 시뮬레이션
        log.info("앙상블 거래 시뮬레이션 시작...")
        trades = simulate_trading_ensemble(valid_prices, predictions, confidences, 0.2)
        
        # 성과 분석
        performance = analyze_performance(trades, valid_prices)
        performance['period'] = period_info['period']
        performance['start_time'] = period_info['start_time']
        performance['end_time'] = period_info['end_time']
        performance['data_points'] = len(valid_prices)
        performance['avg_confidence'] = np.mean(confidences)
        
        results.append(performance)
        
        log.info(f"기간 {period_info['period']} 완료: {performance['total_return']*100:+.2f}%")
    
    # 전체 결과 종합
    if results:
        summarize_ensemble_results(results)
    else:
        log.error("모든 기간에서 백테스트 실패")

def simulate_trading_ensemble(prices, predictions, confidences, confidence_threshold):
    """앙상블 모델을 사용한 거래 시뮬레이션"""
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
    
    for i, (price, pred, conf) in enumerate(zip(prices, predictions, confidences)):
        # 예측값 처리
        if isinstance(pred, (list, tuple, np.ndarray)) and len(pred) >= 3:
            q10, q50, q90 = float(pred[0]), float(pred[1]), float(pred[2])
        else:
            q50 = float(pred)
            q10 = q50 * 0.99
            q90 = q50 * 1.01
        
        # 앙상블 신뢰도 사용
        confidence = conf
        
        # 예측 방향
        expected_return = (q50 - price) / price
        direction = 1 if expected_return > 0 else -1
        
        # 포지션 진입 (앙상블 신뢰도 기반)
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
            holding_time = i - entry_idx if entry_idx is not None else 0
            current_pnl = (price - entry) / entry * side
            
            # 최고 수익 업데이트
            if current_pnl > (peak_pnl or 0):
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
            elif trailing_activated and peak_pnl is not None and peak_pnl > 0:
                drawdown_from_peak = (peak_pnl - current_pnl) / peak_pnl
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

def summarize_ensemble_results(results):
    """앙상블 백테스트 결과 종합"""
    print("\n" + "="*80)
    print("🎯 수정된 앙상블 모델 랜덤 백테스트 결과 종합")
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
        print(f"   평균 신뢰도: {result.get('avg_confidence', 0):.3f}")
    
    # 전체 평균
    if results:
        avg_return = np.mean([r['total_return'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        avg_confidence = np.mean([r.get('avg_confidence', 0) for r in results])
        total_trades = sum([r['total_trades'] for r in results])
        
        print(f"\n🏆 전체 평균 성과:")
        print(f"   평균 수익률: {avg_return*100:+.2f}%")
        print(f"   평균 승률: {avg_win_rate*100:5.1f}%")
        print(f"   평균 Sharpe: {avg_sharpe:6.2f}")
        print(f"   평균 Profit Factor: {avg_profit_factor:5.2f}")
        print(f"   평균 Max Drawdown: {avg_drawdown*100:+.1f}%")
        print(f"   평균 신뢰도: {avg_confidence:.3f}")
        print(f"   총 거래 수: {total_trades}건")
        
        # 기존 모델과 비교
        print(f"\n📈 앙상블 모델 효과:")
        if avg_win_rate > 0.417:
            print(f"   ✅ 승률 향상: 41.7% → {avg_win_rate*100:.1f}%")
        else:
            print(f"   ⚠️ 승률 변화: 41.7% → {avg_win_rate*100:.1f}%")
        
        if avg_return > -0.0025:
            print(f"   ✅ 수익률 향상: -0.25% → {avg_return*100:+.2f}%")
        else:
            print(f"   ⚠️ 수익률 변화: -0.25% → {avg_return*100:+.2f}%")
        
        print(f"\n🌳 전체 10년 데이터로 훈련된 RandomForest 앙상블 적용 완료!")
    
    print("="*80)

if __name__ == "__main__":
    ensemble_backtest_fixed()
