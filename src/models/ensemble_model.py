"""
앙상블 모델 - 여러 모델의 예측을 조합하여 정확도 향상
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class EnsembleModel:
    """앙상블 모델 - 여러 모델의 예측을 조합"""
    
    def __init__(self, 
                 nhits_model,
                 confidence_threshold: float = 0.2,
                 ensemble_method: str = 'weighted_vote'):
        """
        Args:
            nhits_model: 학습된 NHiTS 모델
            confidence_threshold: 신뢰도 임계값
            ensemble_method: 앙상블 방법 ('weighted_vote', 'simple_vote', 'stacking')
        """
        self.nhits_model = nhits_model
        self.confidence_threshold = confidence_threshold
        self.ensemble_method = ensemble_method
        
        # 앙상블 가중치 (초기값)
        self.weights = {
            'nhits': 0.6,      # NHiTS 모델 가중치
            'technical': 0.3,  # 기술적 지표 가중치
            'regime': 0.1      # 시장 상황 가중치
        }
        
        # 기술적 지표 모델
        self.technical_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 시장 상황 분류기
        self.regime_classifier = None
        
        # 스케일러
        self.scaler = StandardScaler()
        
        # 모델 훈련 상태
        self.is_trained = False
        
    def train_technical_model(self, features: pd.DataFrame, targets: np.ndarray):
        """기술적 지표 모델 훈련"""
        try:
            print(f"기술적 지표 모델 훈련 시작:")
            print(f"  - 특징 데이터: {features.shape}")
            print(f"  - 타겟 데이터: {targets.shape}")
            
            # 데이터 길이 확인 및 조정
            min_length = min(len(features), len(targets))
            if min_length < 100:
                print(f"❌ 데이터가 너무 적습니다: {min_length} < 100")
                return False
            
            # 데이터 길이 맞춤 (인덱스 리셋)
            features_aligned = features.iloc[:min_length].reset_index(drop=True)
            targets_aligned = targets[:min_length]
            
            print(f"  - 조정된 특징: {features_aligned.shape}")
            print(f"  - 조정된 타겟: {targets_aligned.shape}")
            
            # NaN 값 제거
            features_clean = features_aligned.dropna()
            if len(features_clean) == 0:
                print("❌ 특징 데이터에 유효한 값이 없습니다")
                return False
            
            # 타겟도 같은 인덱스로 정렬 (인덱스 리셋 후)
            targets_clean = targets_aligned[features_clean.index]
            
            print(f"  - 최종 특징: {features_clean.shape}")
            print(f"  - 최종 타겟: {targets_clean.shape}")
            
            # 특징 스케일링
            features_scaled = self.scaler.fit_transform(features_clean)
            
            # 모델 훈련
            self.technical_model.fit(features_scaled, targets_clean)
            
            print("✅ 기술적 지표 모델 훈련 완료")
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"❌ 기술적 지표 모델 훈련 실패: {e}")
            return False
    
    def predict_regime(self, features: pd.DataFrame) -> np.ndarray:
        """시장 상황 분류 (간단한 규칙 기반)"""
        try:
            # 간단한 규칙 기반 시장 상황 분류
            regimes = []
            
            for _, row in features.iterrows():
                # 변동성 기반 분류
                volatility = row.get('volatility', 0.02)
                
                # 추세 기반 분류
                trend = row.get('trend', 0)
                
                if volatility > 0.05:
                    regime = 'high_volatility'
                elif trend > 0.01:
                    regime = 'bull_market'
                elif trend < -0.01:
                    regime = 'bear_market'
                else:
                    regime = 'sideways'
                
                regimes.append(regime)
            
            return np.array(regimes)
            
        except Exception as e:
            print(f"❌ 시장 상황 분류 실패: {e}")
            return np.array(['sideways'] * len(features))
    
    def predict_technical(self, features: pd.DataFrame) -> np.ndarray:
        """기술적 지표 기반 예측"""
        try:
            if not self.is_trained:
                print("⚠️ 기술적 지표 모델이 훈련되지 않았습니다")
                return np.zeros(len(features))
            
            # NaN 값 처리
            features_clean = features.dropna()
            if len(features_clean) == 0:
                print("⚠️ 특징 데이터에 유효한 값이 없습니다")
                return np.zeros(len(features))
            
            # 특징 스케일링
            features_scaled = self.scaler.transform(features_clean)
            
            # 예측
            predictions = self.technical_model.predict(features_scaled)
            
            # 원본 길이에 맞춰 반환 (NaN 위치는 0으로)
            result = np.zeros(len(features))
            result[features_clean.index] = predictions
            
            return result
            
        except Exception as e:
            print(f"❌ 기술적 지표 예측 실패: {e}")
            return np.zeros(len(features))
    
    def ensemble_predict(self, 
                        nhits_features: np.ndarray,
                        technical_features: pd.DataFrame,
                        current_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        앙상블 예측 실행
        
        Returns:
            tuple: (ensemble_predictions, ensemble_confidences)
        """
        try:
            # 1. NHiTS 모델 예측
            nhits_predictions = self.nhits_model.predict(nhits_features)
            
            # 2. 기술적 지표 예측
            technical_predictions = self.predict_technical(technical_features)
            
            # 3. 시장 상황 분류
            regime_predictions = self.predict_regime(technical_features)
            
            # 4. 앙상블 예측 조합
            if self.ensemble_method == 'weighted_vote':
                ensemble_preds, ensemble_confs = self._weighted_vote(
                    nhits_predictions, technical_predictions, regime_predictions, current_prices
                )
            elif self.ensemble_method == 'simple_vote':
                ensemble_preds, ensemble_confs = self._simple_vote(
                    nhits_predictions, technical_predictions, regime_predictions, current_prices
                )
            else:
                ensemble_preds, ensemble_confs = self._stacking(
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
            tech_pred = technical_preds[i] if i < len(technical_preds) else 0
            
            # 시장 상황에 따른 조정
            regime = regime_preds[i] if i < len(regime_preds) else 'sideways'
            
            # 가중 평균 계산
            if regime == 'high_volatility':
                # 고변동성 시장에서는 기술적 지표 가중치 증가
                weights = {'nhits': 0.4, 'technical': 0.5, 'regime': 0.1}
            elif regime == 'bull_market':
                # 상승장에서는 NHiTS 가중치 증가
                weights = {'nhits': 0.7, 'technical': 0.2, 'regime': 0.1}
            elif regime == 'bear_market':
                # 하락장에서는 기술적 지표 가중치 증가
                weights = {'nhits': 0.5, 'technical': 0.4, 'regime': 0.1}
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
            predictions = [nhits_q50, tech_pred, current_prices[i] * 1.001]
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
            tech_pred = technical_preds[i] if i < len(technical_preds) else 0
            
            # 단순 평균
            ensemble_pred = (nhits_q50 + tech_pred) / 2
            
            # 신뢰도 (두 예측값의 일관성)
            confidence = 1.0 / (1.0 + abs(nhits_q50 - tech_pred) / current_prices[i])
            
            ensemble_predictions.append(ensemble_pred)
            ensemble_confidences.append(confidence)
        
        return np.array(ensemble_predictions), np.array(ensemble_confidences)
    
    def _stacking(self, 
                   nhits_preds: np.ndarray,
                   technical_preds: np.ndarray,
                   regime_preds: np.ndarray,
                   current_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """스태킹 방식 앙상블 (현재는 단순화)"""
        # 스태킹은 복잡하므로 현재는 가중 투표와 동일하게 처리
        return self._weighted_vote(nhits_preds, technical_preds, regime_preds, current_prices)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """앙상블 가중치 업데이트"""
        self.weights.update(new_weights)
        print(f"✅ 앙상블 가중치 업데이트: {self.weights}")
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'is_trained': self.is_trained,
            'confidence_threshold': self.confidence_threshold
        }
