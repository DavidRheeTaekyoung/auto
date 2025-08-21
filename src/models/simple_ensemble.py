"""
간단한 앙상블 모델 - 훈련 없이 바로 사용
여러 예측을 조합하여 정확도 향상
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

class SimpleEnsemble:
    """훈련 없이 바로 사용하는 간단한 앙상블 모델"""
    
    def __init__(self, 
                 nhits_model,
                 confidence_threshold: float = 0.2,
                 ensemble_method: str = 'weighted_vote'):
        """
        Args:
            nhits_model: 학습된 NHiTS 모델
            confidence_threshold: 신뢰도 임계값
            ensemble_method: 앙상블 방법 ('weighted_vote', 'simple_vote')
        """
        self.nhits_model = nhits_model
        self.confidence_threshold = confidence_threshold
        self.ensemble_method = ensemble_method
        
        # 앙상블 가중치 (고정값)
        self.weights = {
            'nhits': 0.7,      # NHiTS 모델 가중치 (높음)
            'technical': 0.2,  # 기술적 지표 가중치 (낮음)
            'regime': 0.1      # 시장 상황 가중치 (매우 낮음)
        }
    
    def predict_technical_simple(self, prices: np.ndarray) -> np.ndarray:
        """간단한 기술적 지표 예측 (훈련 없음)"""
        predictions = []
        
        for i in range(len(prices)):
            if i < 20:  # 충분한 데이터가 없으면 현재 가격
                pred = prices[i]
            else:
                # 간단한 이동평균 기반 예측
                ma_short = np.mean(prices[max(0, i-5):i+1])
                ma_long = np.mean(prices[max(0, i-20):i+1])
                
                # 추세 방향
                trend = (ma_short - ma_long) / ma_long
                
                # 현재 가격에서 추세만큼 조정
                pred = prices[i] * (1 + trend * 0.1)
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_regime_simple(self, prices: np.ndarray) -> np.ndarray:
        """간단한 시장 상황 분류 (훈련 없음)"""
        regimes = []
        
        for i in range(len(prices)):
            if i < 20:
                regime = 'sideways'
            else:
                # 변동성 계산
                recent_prices = prices[max(0, i-20):i+1]
                volatility = np.std(recent_prices) / np.mean(recent_prices)
                
                # 추세 계산
                if i >= 50:
                    trend = (prices[i] - prices[i-50]) / prices[i-50]
                else:
                    trend = 0
                
                # 규칙 기반 분류
                if volatility > 0.03:  # 3% 이상 변동성
                    regime = 'high_volatility'
                elif trend > 0.02:  # 2% 이상 상승
                    regime = 'bull_market'
                elif trend < -0.02:  # 2% 이상 하락
                    regime = 'bear_market'
                else:
                    regime = 'sideways'
            
            regimes.append(regime)
        
        return np.array(regimes)
    
    def ensemble_predict(self, 
                        nhits_features: np.ndarray,
                        current_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        앙상블 예측 실행 (훈련 없음)
        
        Returns:
            tuple: (ensemble_predictions, ensemble_confidences)
        """
        try:
            # 1. NHiTS 모델 예측
            nhits_predictions = self.nhits_model.predict(nhits_features)
            
            # 2. 간단한 기술적 지표 예측
            technical_predictions = self.predict_technical_simple(current_prices)
            
            # 3. 간단한 시장 상황 분류
            regime_predictions = self.predict_regime_simple(current_prices)
            
            # 4. 앙상블 예측 조합
            if self.ensemble_method == 'weighted_vote':
                ensemble_preds, ensemble_confs = self._weighted_vote(
                    nhits_predictions, technical_predictions, regime_predictions, current_prices
                )
            else:
                ensemble_preds, ensemble_confs = self._simple_vote(
                    nhits_predictions, technical_predictions, regime_predictions, current_prices
                )
            
            return ensemble_preds, ensemble_confs
            
        except Exception as e:
            print(f"❌ 앙상블 예측 실패: {e}")
            # 실패 시 NHiTS 예측만 반환
            return nhits_predictions, np.ones(len(nhits_predictions)) * 0.5
    
    def _weighted_vote(self, 
                       nhits_preds: np.ndarray,
                       technical_preds: np.ndarray,
                       regime_preds: np.ndarray,
                       current_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """가중 투표 방식 앙상블"""
        
        ensemble_predictions = []
        ensemble_confidences = []
        
        for i in range(len(nhits_preds)):
            # NHiTS 예측값 처리
            if isinstance(nhits_preds[i], (list, tuple, np.ndarray)) and len(nhits_preds[i]) >= 3:
                nhits_q50 = float(nhits_preds[i][1])  # 중간값
            else:
                nhits_q50 = float(nhits_preds[i])
            
            # 기술적 지표 예측값
            tech_pred = technical_preds[i] if i < len(technical_preds) else current_prices[i]
            
            # 시장 상황에 따른 조정
            regime = regime_preds[i] if i < len(regime_preds) else 'sideways'
            
            # 가중 평균 계산
            if regime == 'high_volatility':
                # 고변동성 시장에서는 기술적 지표 가중치 증가
                weights = {'nhits': 0.5, 'technical': 0.4, 'regime': 0.1}
            elif regime == 'bull_market':
                # 상승장에서는 NHiTS 가중치 증가
                weights = {'nhits': 0.8, 'technical': 0.1, 'regime': 0.1}
            elif regime == 'bear_market':
                # 하락장에서는 기술적 지표 가중치 증가
                weights = {'nhits': 0.6, 'technical': 0.3, 'regime': 0.1}
            else:
                # 횡보장에서는 기본 가중치
                weights = self.weights
            
            # 가중 평균 예측값
            ensemble_pred = (
                weights['nhits'] * nhits_q50 +
                weights['technical'] * tech_pred +
                weights['regime'] * current_prices[i] * 0.001  # 시장 상황 영향
            )
            
            # 신뢰도 계산 (예측값들의 일관성 기반)
            predictions = [nhits_q50, tech_pred, current_prices[i]]
            confidence = 1.0 / (1.0 + np.std(predictions) / current_prices[i])
            
            ensemble_predictions.append(ensemble_pred)
            ensemble_confidences.append(confidence)
        
        return np.array(ensemble_predictions), np.array(ensemble_confidences)
    
    def _simple_vote(self, 
                     nhits_preds: np.ndarray,
                     technical_preds: np.ndarray,
                     regime_preds: np.ndarray,
                     current_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """단순 투표 방식 앙상블"""
        
        ensemble_predictions = []
        ensemble_confidences = []
        
        for i in range(len(nhits_preds)):
            # NHiTS 예측값
            if isinstance(nhits_preds[i], (list, tuple, np.ndarray)) and len(nhits_preds[i]) >= 3:
                nhits_q50 = float(nhits_preds[i][1])
            else:
                nhits_q50 = float(nhits_preds[i])
            
            # 기술적 지표 예측값
            tech_pred = technical_preds[i] if i < len(technical_preds) else current_prices[i]
            
            # 단순 평균
            ensemble_pred = (nhits_q50 + tech_pred) / 2
            
            # 신뢰도 (두 예측값의 일관성)
            confidence = 1.0 / (1.0 + abs(nhits_q50 - tech_pred) / current_prices[i])
            
            ensemble_predictions.append(ensemble_pred)
            ensemble_confidences.append(confidence)
        
        return np.array(ensemble_predictions), np.array(ensemble_confidences)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """앙상블 가중치 업데이트"""
        self.weights.update(new_weights)
        print(f"✅ 앙상블 가중치 업데이트: {self.weights}")
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'confidence_threshold': self.confidence_threshold,
            'note': '훈련 없이 바로 사용 가능한 간단한 앙상블'
        }
