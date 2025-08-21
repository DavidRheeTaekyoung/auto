"""
실전 하이퍼파라미터 최적화 스크립트
빠르고 안정적인 최적화를 위한 실용적 접근법
"""

import pickle
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset
from src.models.nhits_model import Predictor

class PracticalOptimizer:
    """실전적 최적화 - 빠르고 안정적"""
    
    def __init__(self, logger):
        self.logger = logger
        # 검증된 좋은 시작점들 (경험적 지식)
        self.known_good_configs = [
            {
                'lr': 3e-4,
                'batch_size': 128,
                'input_window': 512,
                'epochs': 50,
                'confidence_threshold': 0.3,  # 더 낮춰서 거래 발생하도록
                'train_months': 6
            },
            {
                'lr': 1e-3,
                'batch_size': 256,
                'input_window': 256,
                'epochs': 40,
                'confidence_threshold': 0.4,
                'train_months': 4
            },
            {
                'lr': 5e-4,
                'batch_size': 64,
                'input_window': 512,
                'epochs': 30,
                'confidence_threshold': 0.2,  # 더 낮게
                'train_months': 3
            }
        ]
    
    def evaluate_config(self, config, data):
        """설정 평가 - 거래 발생 여부 확인 포함"""
        try:
            self.logger.info(f"설정 평가: {config}")
            
            # 데이터 준비
            max_rows = config['train_months'] * 30 * 24 * 60
            data_subset = data.tail(min(max_rows, len(data)))
            
            # 특징 생성
            feat = enrich(data_subset)
            if len(feat) < config['input_window'] + 1000:
                self.logger.warning("데이터 부족")
                return -999.0
            
            # 데이터셋 생성
            X, Y = make_dataset(feat, config['input_window'], (60, 240, 1440))
            if len(X) < 1000:
                return -999.0
            
            # Train/Valid 분할
            split_idx = int(len(X) * 0.8)
            X_train, Y_train = X[:split_idx], Y[:split_idx]
            X_valid, Y_valid = X[split_idx:], Y[split_idx:]
            
            # 모델 학습
            model = Predictor(
                in_feat=X.shape[2],
                horizons=(60, 240, 1440),
                quantiles=(0.1, 0.5, 0.9),
                lr=config['lr']
            )
            
            # 빠른 학습
            model.fit(X_train, Y_train, epochs=min(config['epochs'], 20), batch=config['batch_size'])
            
            # 예측
            predictions = []
            chunk_size = 500
            for i in range(0, len(X_valid), chunk_size):
                end = min(i + chunk_size, len(X_valid))
                batch_pred = model.predict(X_valid[i:end])
                for pred in batch_pred:
                    predictions.append(pred[0])  # 1시간 예측값
            
            # 가격 데이터
            valid_start = config['input_window'] + split_idx
            valid_prices = feat["close"].iloc[valid_start:valid_start+len(predictions)].values
            
            # 거래 시뮬레이션
            trades = self.simulate_trading(valid_prices, predictions, config['confidence_threshold'])
            
            # ===== 중요: 거래 발생 확인 =====
            if len(trades) == 0:
                self.logger.warning(f"거래 없음! threshold={config['confidence_threshold']}가 너무 높음")
                return -2.0
            
            self.logger.info(f"거래 발생: {len(trades)}건")
            
            # 성과 계산
            score = self.calculate_score(trades)
            
            self.logger.info(f"최종 점수: {score:.3f}")
            return score
            
        except Exception as e:
            self.logger.error(f"설정 평가 중 오류: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return -999.0
    
    def simulate_trading(self, prices, predictions, confidence_threshold):
        """거래 시뮬레이션 - Trailing Stop 적용"""
        trades = []
        cash = 10000
        pos_qty = 0.0
        entry = None
        side = None
        entry_idx = None
        
        # ===== Trailing Stop 변수 추가 =====
        peak_price = None  # 최고가 추적
        peak_pnl = None    # 최고 수익률 추적
        trailing_activated = False  # Trailing Stop 활성화 여부
        
        for i, (price, pred) in enumerate(zip(prices, predictions)):
            # 예측값 처리 (더 안전하게)
            if isinstance(pred, dict):
                # dict 형태인 경우
                q10 = pred.get(0.1, price * 0.99)
                q50 = pred.get(0.5, price)
                q90 = pred.get(0.9, price * 1.01)
            elif isinstance(pred, (list, tuple, np.ndarray)):
                if len(pred) >= 3:
                    q10, q50, q90 = float(pred[0]), float(pred[1]), float(pred[2])
                else:
                    q50 = float(pred[0]) if len(pred) > 0 else price
                    q10 = q50 * 0.99
                    q90 = q50 * 1.01
            else:
                q50 = float(pred)
                q10 = q50 * 0.99
                q90 = q50 * 1.01
            
            # ===== 수정된 신뢰도 계산 =====
            # 예측 범위가 넓을수록 불확실 → 낮은 신뢰도
            pred_range = abs(q90 - q10)
            confidence = 1.0 / (1.0 + pred_range / price * 100)  # 정규화된 신뢰도
            
            # 예측 방향과 강도
            expected_return = (q50 - price) / price
            direction = 1 if expected_return > 0 else -1
            
            # ===== 포지션 진입 (더 현실적) =====
            if pos_qty == 0:
                # 신뢰도와 기대수익률 모두 고려
                if confidence > confidence_threshold and abs(expected_return) > 0.001:  # 0.1% 이상
                    pos_qty = (cash * 2) / price  # 레버리지 낮춤
                    side = direction
                    entry = price
                    entry_idx = i
                    
                    # Trailing Stop 초기화
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
            
            # ===== 포지션 관리 (Trailing Stop) =====
            elif pos_qty != 0:
                holding_time = i - entry_idx
                current_pnl = (price - entry) / entry * side
                
                # 최고 수익 업데이트
                if current_pnl > peak_pnl:
                    peak_pnl = current_pnl
                    peak_price = price
                    
                    # 1% 이상 수익 시 Trailing Stop 활성화
                    if peak_pnl >= 0.01:
                        trailing_activated = True
                
                # 청산 조건들
                should_exit = False
                exit_reason = ""
                
                # 1. 손절 (더 관대하게)
                if current_pnl <= -0.02:  # 2% 손실
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # ===== 2. Trailing Stop (핵심 로직) =====
                elif trailing_activated:
                    # 최고점 대비 하락률 계산
                    drawdown_from_peak = (peak_pnl - current_pnl) / peak_pnl if peak_pnl > 0 else 0
                    
                    # 이익의 30% 이상 반납 시 청산
                    if drawdown_from_peak >= 0.3:
                        should_exit = True
                        exit_reason = f"trailing_stop (peak: {peak_pnl:.1%}, current: {current_pnl:.1%})"
                        
                    # 또는 절대 수익률이 0.7% 아래로 떨어지면 청산 (1%의 70%)
                    elif peak_pnl >= 0.01 and current_pnl < peak_pnl * 0.7:
                        should_exit = True
                        exit_reason = "profit_protection"
                
                # 3. 시간 기반 청산 (선택적)
                elif holding_time >= 120:  # 2시간
                    should_exit = True
                    exit_reason = "timeout"
                
                # 4. 강한 반대 신호
                elif confidence > confidence_threshold * 1.2 and expected_return * side < -0.002:
                    should_exit = True
                    exit_reason = "strong_reverse_signal"
                
                # ===== 5. 목표 수익 도달 (큰 수익 허용) =====
                elif current_pnl >= 0.05:  # 5% 이상은 무조건 청산
                    should_exit = True
                    exit_reason = "target_reached"
                
                if should_exit:
                    cash += pos_qty * (price - entry) * side
                    trades.append({
                        'type': 'exit',
                        'price': price,
                        'pnl_pct': current_pnl,
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
            trades.append({
                'type': 'exit',
                'price': final_price,
                'pnl_pct': final_pnl,
                'peak_pnl': peak_pnl,
                'reason': 'force_close',
                'time': len(prices) - 1
            })
        
        return trades
    
    def calculate_score(self, trades):
        """복합 점수 계산 - Trailing Stop 거래에 최적화"""
        if len(trades) < 2:
            return -1.0
        
        # 거래별 수익률 계산
        returns = []
        trade_pairs = []
        
        # entry-exit 쌍 찾기
        for i in range(len(trades)):
            if trades[i]['type'] == 'entry':
                # 다음 exit 찾기
                for j in range(i + 1, len(trades)):
                    if trades[j]['type'] == 'exit':
                        trade_pairs.append((trades[i], trades[j]))
                        returns.append(trades[j]['pnl_pct'])
                        break
        
        # ===== 디버깅 정보 추가 =====
        self.logger.debug(f"총 거래 신호: {len(trades)}")
        self.logger.debug(f"완료된 거래 쌍: {len(trade_pairs)}")
        
        if len(returns) == 0:
            self.logger.warning("완료된 거래가 없음 (entry만 있고 exit 없음)")
            return -1.0
        
        # 기본 메트릭
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.01
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        # 추가 메트릭
        total_return = np.prod([1 + r for r in returns]) - 1
        max_drawdown = np.min(returns) if returns else 0
        
        # Sharpe Ratio (연율화)
        if std_return > 1e-6:
            # 분당 거래 가정
            sharpe = mean_return / std_return * np.sqrt(365 * 24 * 60 / len(returns))
        else:
            sharpe = 0.0
        
        # Profit Factor
        wins = sum([r for r in returns if r > 0])
        losses = sum([abs(r) for r in returns if r < 0])
        profit_factor = wins / losses if losses > 0 else 1.0
        
        # ===== 디버깅 정보 =====
        self.logger.debug(f"평균 수익률: {mean_return:.4f}")
        self.logger.debug(f"승률: {win_rate:.2%}")
        self.logger.debug(f"Sharpe: {sharpe:.3f}")
        
        # 복합 점수 (더 균형잡힌 가중치)
        score = (
            sharpe * 0.3 +
            win_rate * 0.3 +
            profit_factor * 0.2 +
            (1 + max_drawdown) * 0.1 +  # drawdown 페널티
            min(len(returns), 50) / 50 * 0.1  # 거래 빈도 보너스
        )
        
        return score
    
    def optimize(self, data):
        """실용적 최적화 프로세스"""
        
        # 1. 알려진 좋은 설정들 테스트
        self.logger.info("Step 1: Testing known good configurations...")
        best_config, best_score = self.test_known_configs(data)
        
        # 2. Learning Rate 미세조정 (가장 중요)
        self.logger.info("Step 2: Fine-tuning learning rate...")
        best_config['lr'] = self.optimize_lr(best_config, data)
        
        # 3. Confidence Threshold 조정 (거래 빈도)
        self.logger.info("Step 3: Optimizing confidence threshold...")
        best_config['confidence_threshold'] = self.optimize_threshold(best_config, data)
        
        # 4. 최종 검증
        self.logger.info("Step 4: Final validation...")
        final_score = self.validate_final(best_config, data)
        
        return best_config, final_score
    
    def test_known_configs(self, data):
        """검증된 설정 테스트"""
        best_score = -np.inf
        best_config = None
        
        for config in self.known_good_configs:
            # 빠른 테스트 (최근 2개월)
            test_data = data[-60*24*60:]
            score = self.evaluate_config(config, test_data)
            self.logger.info(f"Config score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
        
        return best_config, best_score
    
    def optimize_lr(self, config, data):
        """Learning Rate 최적화"""
        base_lr = config['lr']
        
        # 로그 스케일로 탐색
        lr_candidates = [
            base_lr * 0.3,
            base_lr * 0.5,
            base_lr,
            base_lr * 2,
            base_lr * 3
        ]
        
        best_lr = base_lr
        best_score = -np.inf
        
        for lr in lr_candidates:
            test_config = config.copy()
            test_config['lr'] = lr
            
            test_data = data[-30*24*60:]  # 1개월
            score = self.evaluate_config(test_config, test_data)
            
            if score > best_score:
                best_score = score
                best_lr = lr
        
        self.logger.info(f"Best LR: {best_lr}")
        return best_lr
    
    def optimize_threshold(self, config, data):
        """거래 임계값 최적화"""
        thresholds = np.linspace(0.05, 0.4, 8)  # 더 낮은 범위
        
        best_threshold = 0.2
        best_score = -np.inf
        
        for threshold in thresholds:
            test_config = config.copy()
            test_config['confidence_threshold'] = threshold
            
            # 거래 빈도 체크
            test_data = data[-30*24*60:]  # 1개월
            trades = self.count_trades(test_config, test_data)
            
            # 너무 적거나 많으면 패널티
            if trades < 5 or trades > 500:
                continue
            
            score = self.evaluate_config(test_config, test_data)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.logger.info(f"Best threshold: {best_threshold}")
        return best_threshold
    
    def count_trades(self, config, data):
        """거래 수 계산 (빠른 체크)"""
        try:
            # 간단한 거래 시뮬레이션
            max_rows = config['train_months'] * 30 * 24 * 60
            data_subset = data.tail(min(max_rows, len(data)))
            
            feat = enrich(data_subset)
            X, Y = make_dataset(feat, config['input_window'], (60, 240, 1440))
            
            if len(X) < 100:
                return 0
            
            # 간단한 예측 (모델 학습 없이)
            predictions = np.random.normal(0, 1, len(X[-100:]))  # 랜덤 예측으로 거래 수만 체크
            
            # 거래 시뮬레이션
            trades = self.simulate_trading(
                data_subset['close'].iloc[-100:].values,
                predictions,
                config['confidence_threshold']
            )
            
            return len(trades)
            
        except:
            return 0
    
    def validate_final(self, config, data):
        """최종 검증"""
        self.logger.info("최종 검증 중...")
        final_score = self.evaluate_config(config, data)
        self.logger.info(f"최종 점수: {final_score:.3f}")
        return final_score


if __name__ == "__main__":
    # 설정 로드
    cfg = load_config()
    log = setup_logger(level="DEBUG")  # 디버깅 정보 활성화
    
    log.info("=== 실전 하이퍼파라미터 최적화 시작 ===")
    
    # 1. 데이터 로드
    log.info("데이터베이스에서 데이터 로드 중...")
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    raw = db.load(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"])
    log.info(f"로드된 데이터: {len(raw):,}행")
    
    # 메모리 효율을 위해 최근 6개월 데이터만 사용
    max_data_rows = 6 * 30 * 24 * 60  # 6개월
    if len(raw) > max_data_rows:
        raw = raw.tail(max_data_rows)
        log.info(f"최적화용 데이터: {len(raw):,}행 (6개월)")
    
    # 2. 먼저 디버깅 모드로 1회 실행
    log.info("디버깅 테스트...")
    test_params = {
        'lr': 5e-4,
        'batch_size': 128,
        'input_window': 512,
        'epochs': 30,
        'confidence_threshold': 0.1,  # 낮춰서 거래 발생하도록
        'train_months': 3
    }
    
    optimizer = PracticalOptimizer(log)
    test_score = optimizer.evaluate_config(test_params, raw[-30*24*60:])
    log.info(f"테스트 점수: {test_score}")
    
    if test_score < -900:
        log.error("에러 발생! 코드 확인 필요")
        exit(1)
    elif test_score == -2.0:
        log.warning("거래 없음! Threshold 조정 필요")
    
    # 3. 실전 최적화
    log.info("실전 최적화 시작...")
    best_params, best_score = optimizer.optimize(raw)
    
    log.info("=== 최적화 완료 ===")
    log.info(f"최적 파라미터: {best_params}")
    log.info(f"최종 점수: {best_score:.3f}")
    
    # 4. 결과 저장
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 최적 파라미터 저장
    params_path = os.path.join(models_dir, "best_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(best_params, f)
    
    # JSON으로도 저장
    json_path = os.path.join(models_dir, "best_params.json")
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    log.info(f"최적 파라미터 저장 완료: {params_path}")
    log.info(f"JSON 저장 완료: {json_path}")
    
    print("="*50)
    print("하이퍼파라미터 최적화 완료!")
    print(f"최종 점수: {best_score:.3f}")
    print("최적 파라미터:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*50)
