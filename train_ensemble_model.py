"""
전체 10년 데이터로 앙상블 모델 훈련
RandomForest를 전체 기간으로 훈련하여 저장
"""

import numpy as np
import pandas as pd
import os
import pickle
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset
from src.models.nhits_model import Predictor
from src.models.ensemble_model import EnsembleModel

def create_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 특징 생성"""
    try:
        features = pd.DataFrame()
        
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

def train_ensemble_model():
    """전체 10년 데이터로 앙상블 모델 훈련"""
    
    # 설정 로드
    cfg = load_config()
    log = setup_logger(level="INFO")
    
    log.info("=== 전체 10년 데이터로 앙상블 모델 훈련 시작 ===")
    
    # 전체 데이터 로드
    db = DataStorage(cfg["env"]["DUCKDB_PATH"])
    symbol_db = cfg["data"]["symbol_internal"].replace("/", "")
    raw = db.load(symbol_db, cfg["data"]["timeframe"])
    
    log.info(f"전체 데이터: {len(raw):,}행")
    log.info(f"데이터 기간: {raw['open_time'].min()} ~ {raw['open_time'].max()}")
    
    # 기존 NHiTS 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), "models", "predictor.pkl")
    if not os.path.exists(model_path):
        log.error(f"NHiTS 모델 파일이 없습니다: {model_path}")
        log.info("먼저 train.py를 실행해서 NHiTS 모델을 학습하세요.")
        return
    
    log.info("기존 NHiTS 모델 로드 중...")
    with open(model_path, 'rb') as f:
        nhits_model = pickle.load(f)
    log.info("NHiTS 모델 로드 완료")
    
    # 전체 데이터로 특징 생성
    log.info("전체 데이터로 특징 생성 중...")
    feat = enrich(raw)
    log.info(f"특징 생성 완료: {feat.shape}")
    
    # 앙상블 모델 생성
    log.info("앙상블 모델 생성 중...")
    ensemble = EnsembleModel(
        nhits_model=nhits_model,
        confidence_threshold=0.2,
        ensemble_method='weighted_vote'
    )
    
    # 전체 데이터의 70%를 훈련용으로 사용 (나머지는 백테스트용)
    train_size = int(len(feat) * 0.7)
    train_feat = feat.iloc[:train_size]
    
    log.info(f"RandomForest 훈련용 데이터: {len(train_feat):,}행")
    
    # 기술적 특징 생성 (전체 훈련 데이터)
    log.info("전체 훈련 데이터로 기술적 특징 생성 중...")
    technical_features = create_technical_features(train_feat)
    
    if len(technical_features) == 0:
        log.error("기술적 특징 생성 실패")
        return
    
    log.info(f"기술적 특징 생성 완료: {technical_features.shape}")
    
    # 타겟 데이터 준비 (가격 변화율)
    price_data = train_feat['close']
    price_returns = price_data.pct_change().fillna(0)
    
    # 기술적 특징과 타겟 데이터 길이 맞춤
    min_length = min(len(technical_features), len(price_returns))
    technical_features_aligned = technical_features.iloc[:min_length]
    price_returns_aligned = price_returns.iloc[:min_length]
    
    log.info(f"정렬된 데이터:")
    log.info(f"  - 기술적 특징: {technical_features_aligned.shape}")
    log.info(f"  - 타겟 데이터: {price_returns_aligned.shape}")
    
    # RandomForest 훈련
    log.info("RandomForest 모델 훈련 시작...")
    success = ensemble.train_technical_model(
        technical_features_aligned, 
        price_returns_aligned.values
    )
    
    if not success:
        log.error("RandomForest 훈련 실패")
        return
    
    log.info("✅ RandomForest 훈련 완료!")
    
    # 앙상블 모델 저장
    ensemble_path = os.path.join(os.path.dirname(__file__), "models", "ensemble_model.pkl")
    os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
    
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    log.info(f"✅ 앙상블 모델 저장 완료: {ensemble_path}")
    
    # 모델 정보 출력
    model_info = ensemble.get_model_info()
    log.info(f"앙상블 모델 정보: {model_info}")
    
    # 간단한 테스트
    log.info("앙상블 모델 테스트 중...")
    test_feat = feat.iloc[train_size:train_size+1000]  # 1000개 테스트
    test_technical = create_technical_features(test_feat)
    
    if len(test_technical) > 0:
        test_prices = test_feat['close'].values[:len(test_technical)]
        
        # 간단한 예측 테스트
        try:
            # NHiTS용 데이터셋 생성
            X_test, Y_test = make_dataset(test_feat, 512, (60, 240, 1440))
            if len(X_test) > 0:
                nhits_features = X_test[:10]  # 10개만 테스트
                current_prices = test_prices[:len(nhits_features)]
                tech_features = test_technical.iloc[:len(nhits_features)]
                
                preds, confs = ensemble.ensemble_predict(
                    nhits_features, tech_features, current_prices
                )
                
                log.info(f"✅ 앙상블 예측 테스트 성공!")
                log.info(f"  - 예측값: {len(preds)}개")
                log.info(f"  - 평균 신뢰도: {np.mean(confs):.3f}")
            else:
                log.warning("테스트용 NHiTS 데이터셋 생성 실패")
        except Exception as e:
            log.warning(f"앙상블 예측 테스트 실패: {e}")
    
    log.info("=== 앙상블 모델 훈련 완료 ===")

if __name__ == "__main__":
    train_ensemble_model()
