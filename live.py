import pickle, time, os
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.processors.data_storage import DataStorage
from src.collectors.ohlcv_collector import FuturesCollector
from src.features.feature_maker import enrich, make_dataset
from src.traders.broker_binance import BinanceUSDM
from src.traders.execution_loop import TradeEngine

if __name__=="__main__":
    cfg=load_config(); log=setup_logger(level=cfg["env"]["LOG_LEVEL"])
    
    # 모델 로드 경로를 상대 경로로 설정
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    predictor_path = os.path.join(models_dir, "predictor.pkl")
    
    if not os.path.exists(predictor_path):
        print(f"모델 파일을 찾을 수 없습니다: {predictor_path}")
        print("먼저 run_train.bat을 실행하여 모델을 학습하세요.")
        exit(1)
    
    with open(predictor_path, "rb") as f: pred=pickle.load(f)

    broker=BinanceUSDM(cfg["env"]["BINANCE_API_KEY"], cfg["env"]["BINANCE_SECRET_KEY"], cfg["env"]["TRADING_MODE"])
    coll=FuturesCollector(cfg["env"]["BINANCE_API_KEY"], cfg["env"]["BINANCE_SECRET_KEY"])
    store=DataStorage(cfg["env"]["DUCKDB_PATH"])
    engine=TradeEngine(broker, cfg)

    while True:
        # 1) 최신 1분봉 적재
        last=store.last_timestamp_ms(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"])
        inc=coll.fetch_incremental(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"], last, cfg["data"]["fetch_limit"])
        if not inc.empty: store.upsert_ohlcv(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"], inc)

        # 2) 예측 입력
        df=store.load(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"])
        feat=enrich(df)
        X,_=make_dataset(feat, cfg["model"]["input_window"], tuple(cfg["model"]["horizons"].values()))
        if len(X)==0: time.sleep(cfg["live"]["loop_sec"]); continue
        q=pred.predict(X[-1:])[0]  # [H,Q]
        q10,q50,q90 = q[0]

        # 3) 브로커 현금/중간가
        mid=broker.mid()
        cash=broker.balance()["free"]

        # 4) 진입/관리
        if engine.active is None:
            info=engine.decide_and_execute(price=mid, q10=q10,q50=q50,q90=q90, cash_usdt=cash)
            log.info(f"OPEN side={info['side']} lev={info['lev']:.2f} c={info['c']:.2f} qty={info['qty']:.6f} entry~{info['entry']:.2f}")
        else:
            stat=engine.on_price_tick(mid)
            if stat: log.info(f"RUN pnl={stat['pnl']:.2f} stop~{stat['stop']:.2f}")

        time.sleep(cfg["live"]["loop_sec"])
