import pickle, os
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset
from src.models.nhits_model import Predictor
from src.models.regime_detector import RegimeDetector

if __name__=="__main__":
    cfg=load_config(); log=setup_logger(level=cfg["env"]["LOG_LEVEL"])
    
    log.info("=== 모델 학습 시작 ===")
    
    # 모델 저장 디렉토리 생성
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    log.info(f"모델 저장 디렉토리: {models_dir}")
    
    # 최적화된 파라미터 로드 시도
    best_params_path = os.path.join(models_dir, "best_params.pkl")
    if os.path.exists(best_params_path):
        log.info("최적화된 파라미터 발견! 자동으로 적용합니다.")
        with open(best_params_path, "rb") as f:
            best_params = pickle.load(f)
        log.info(f"최적 파라미터: {best_params}")
        
        # 최적 파라미터 적용
        input_window = best_params.get('input_window', cfg["model"]["input_window"])
        lr = best_params.get('lr', cfg["model"]["lr"])
        epochs = best_params.get('epochs', cfg["model"]["epochs"])
        batch_size = best_params.get('batch_size', cfg["model"]["batch_size"])
        train_months = best_params.get('train_months', 6)
    else:
        log.info("최적화된 파라미터가 없습니다. 기본 설정을 사용합니다.")
        log.info("더 나은 성능을 위해 run_optimize.bat을 먼저 실행하는 것을 권장합니다.")
        
        # 기본 파라미터 사용
        input_window = cfg["model"]["input_window"]
        lr = cfg["model"]["lr"]
        epochs = cfg["model"]["epochs"]
        batch_size = cfg["model"]["batch_size"]
        train_months = 6
    
    # 데이터 로드
    log.info("데이터베이스에서 데이터 로드 중...")
    db=DataStorage(cfg["env"]["DUCKDB_PATH"])
    raw=db.load(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"])
    log.info(f"로드된 데이터: {len(raw):,}행")
    
    if len(raw)<2000: 
        log.error("데이터 부족(>=2000분 필요)")
        raise SystemExit("데이터 부족(>=2000분 필요)")
    
    # 메모리 효율을 위해 최적화된 개월 수 사용
    max_rows = train_months * 30 * 24 * 60
    if len(raw) > max_rows:
        log.info(f"메모리 효율을 위해 최근 {max_rows:,}행({train_months}개월)만 사용합니다")
        raw = raw.tail(max_rows)
        log.info(f"사용할 데이터: {len(raw):,}행")
    
    # 특징 생성
    log.info("기술적 특징 생성 중...")
    feat=enrich(raw)
    log.info(f"특징 생성 완료: {len(feat):,}행")
    
    # 데이터셋 생성 (최적화된 input_window 사용)
    log.info(f"학습 데이터셋 생성 중... (input_window={input_window})")
    X,Y=make_dataset(feat, input_window, tuple(cfg["model"]["horizons"].values()))
    log.info(f"데이터셋 생성 완료: X={X.shape}, Y={Y.shape}")
    
    # 학습/검증 분할
    cut=int(len(X)*0.8); Xtr,Ytr=X[:cut],Y[:cut]
    log.info(f"학습 데이터: {len(Xtr):,}개, 검증 데이터: {len(X)-len(Xtr):,}개")
    
    # NHiTS 모델 학습 (최적화된 파라미터 사용)
    log.info(f"NHiTS 모델 학습 시작... (lr={lr}, epochs={epochs}, batch={batch_size})")
    pred=Predictor(X.shape[2], tuple(cfg["model"]["horizons"].values()),
                   tuple(cfg["model"]["quantiles"]), lr=lr)
    pred.fit(Xtr,Ytr, epochs=epochs, batch=batch_size)
    log.info("NHiTS 모델 학습 완료")
    
    # Regime 모델 학습
    log.info("Regime 모델 학습 시작...")
    reg=RegimeDetector(cfg["regime"]["n_states"]); 
    reg.fit(feat)
    log.info("Regime 모델 학습 완료")
    
    # 모델 저장
    log.info("모델 저장 중...")
    predictor_path = os.path.join(models_dir, "predictor.pkl")
    regime_path = os.path.join(models_dir, "regime.pkl")
    
    with open(predictor_path, "wb") as f: pickle.dump(pred,f)
    with open(regime_path, "wb") as f: pickle.dump(reg,f)
    
    log.info(f"[OK] 모델 저장 완료: {models_dir}")
    print(f"[OK] models saved to {models_dir}")
    
    # 사용된 파라미터 정보 출력
    print(f"사용된 파라미터:")
    print(f"  input_window: {input_window}")
    print(f"  learning_rate: {lr}")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  train_months: {train_months}")
