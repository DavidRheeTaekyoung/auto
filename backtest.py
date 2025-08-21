import pickle, numpy as np, pandas as pd, os
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset
from src.traders.sizing import confidence_from_quantiles
from src.traders.risk import stop_price_for_account_risk, TrailingManager

def backtest_loop(prices, preds_1h, cash0=10000, fee_bp=6, slip_bp=5):
    cash=cash0; pos_qty=0.0; entry=None; side=None
    trail=TrailingManager(0.30); eq_curve=[]

    for i,(p,pred) in enumerate(zip(prices, preds_1h)):
        # 예측값 형식 확인 및 처리
        if isinstance(pred, (list, tuple, np.ndarray)) and len(pred) >= 3:
            # 3개 이상의 값이 있는 경우 (q10, q50, q90)
            q10, q50, q90 = pred[0], pred[1], pred[2]
        else:
            # 단일 값인 경우 (중간값으로 처리)
            q50 = float(pred)
            q10 = q50 * 0.9  # 임시로 10% 아래
            q90 = q50 * 1.1  # 임시로 10% 위
        
        # 포지션 없으면 진입 판단
        if pos_qty==0:
            c,dirn=confidence_from_quantiles(q10,q50,q90)
            if c<0.2: eq_curve.append(cash); continue  # 확신 낮으면 건너뜀
            notional=min(cash*5, cash*25) # 단순: 확신도 없으면 5x, 있으면 25x까지 점증 (학습 외)
            pos_qty=notional/p
            side=1 if dirn>0 else -1
            entry=p
            # 초기 스탑: 계좌 10% 손실
            stop=stop_price_for_account_risk(entry, pos_qty, cash, is_long=(side==1), cash_risk_pct=0.10)
            trail=TrailingManager(0.30)
        else:
            # PnL 업데이트
            pnl = pos_qty * (p - entry) * side
            trail.update(max(pnl,0))
            # 트레일링 청산
            trail_price = trail.trailing_stop_price(entry, pos_qty, is_long=(side==1))
            stop = trail_price if (side==1 and trail_price>entry) or (side==-1 and trail_price<entry) else stop
            # 스탑 체킹
            if (side==1 and p<=stop) or (side==-1 and p>=stop):
                # 청산: 수수료/슬리피지 반영
                fee = abs(pos_qty*entry)*(fee_bp/1e4)
                slip= abs(pos_qty*(entry-p))* (slip_bp/1e4)
                cash += pnl - fee - slip
                pos_qty=0; entry=None; side=None
        eq_curve.append(cash + (pos_qty*(p-entry)*side if pos_qty!=0 else 0))
    return pd.Series(eq_curve)

if __name__=="__main__":
    cfg=load_config(); log=setup_logger(level=cfg["env"]["LOG_LEVEL"])
    
    log.info("=== 백테스트 시작 ===")
    
    # 모델 로드 경로를 상대 경로로 설정
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    predictor_path = os.path.join(models_dir, "predictor.pkl")
    
    if not os.path.exists(predictor_path):
        log.error(f"모델 파일을 찾을 수 없습니다: {predictor_path}")
        print(f"모델 파일을 찾을 수 없습니다: {predictor_path}")
        print("먼저 run_train.bat을 실행하여 모델을 학습하세요.")
        exit(1)
    
    log.info("학습된 모델 로드 중...")
    with open(predictor_path, "rb") as f: pred=pickle.load(f)
    log.info("모델 로드 완료")
    
    # 데이터 로드 (메모리 효율을 위해 최근 데이터만)
    log.info("데이터베이스에서 데이터 로드 중...")
    db=DataStorage(cfg["env"]["DUCKDB_PATH"])
    raw=db.load(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"])
    log.info(f"로드된 데이터: {len(raw):,}행")
    
    # 메모리 효율을 위해 최근 데이터만 사용 (최근 3개월)
    max_rows = 3 * 30 * 24 * 60  # 3개월 * 30일 * 24시간 * 60분
    if len(raw) > max_rows:
        log.info(f"메모리 효율을 위해 최근 {max_rows:,}행만 사용합니다")
        raw = raw.tail(max_rows)
        log.info(f"사용할 데이터: {len(raw):,}행")
    
    # 특징 생성
    log.info("기술적 특징 생성 중...")
    feat=enrich(raw)
    log.info(f"특징 생성 완료: {len(feat):,}행")
    
    # 데이터셋 생성
    log.info("백테스트용 데이터셋 생성 중...")
    X,Y=make_dataset(feat, cfg["model"]["input_window"], tuple(cfg["model"]["horizons"].values()))
    log.info(f"데이터셋 생성 완료: X={X.shape}, Y={Y.shape}")
    
    # 백테스트 시작점 (학습 데이터 이후)
    start=int(len(X)*0.8)
    log.info(f"백테스트 시작점: {start:,}개 데이터 이후")
    
    # 예측 생성 (청크 단위로 처리)
    log.info("예측 생성 중...")
    preds=[]
    chunk_size = 1000  # 메모리 효율을 위해 1000개씩 처리
    
    for i in range(start, len(X), chunk_size):
        end = min(i + chunk_size, len(X))
        log.info(f"예측 진행률: {i-start:,}/{len(X)-start:,} ({((i-start)/(len(X)-start)*100):.1f}%)")
        
        for j in range(i, end):
            out=pred.predict(X[j:j+1])[0]  # [H,Q]
            preds.append(out[0])           # 1h 사용
    
    log.info(f"예측 완료: {len(preds):,}개")
    
    # 가격 데이터 준비
    prices=feat["close"].iloc[cfg["model"]["input_window"]+start : cfg["model"]["input_window"]+start+len(preds)].values
    log.info(f"가격 데이터 준비 완료: {len(prices):,}개")
    
    # 백테스트 실행
    log.info("백테스트 루프 실행 중...")
    eq=backtest_loop(prices, preds, cash0=10000, fee_bp=cfg["trade"]["fee_bp"], slip_bp=cfg["trade"]["slip_bp"])
    
    # 성과 계산
    ret=eq.pct_change().fillna(0)
    sharpe=(ret.mean()/(ret.std()+1e-9))*np.sqrt(365*24*60)  # 1m 기준 근사
    dd=(eq/eq.cummax()-1).min()
    
    log.info("=== 백테스트 결과 ===")
    log.info(f"Sharpe Ratio: {sharpe:.3f}")
    log.info(f"Maximum Drawdown: {dd:.3f}")
    log.info(f"Final Equity: ${eq.iloc[-1]:,.2f}")
    
    result = {"sharpe":float(sharpe), "max_dd":float(dd), "final_eq":float(eq.iloc[-1])}
    print(result)
