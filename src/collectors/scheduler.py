import schedule, time
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.collectors.ohlcv_collector import FuturesCollector
from src.processors.data_storage import DataStorage
from datetime import datetime, timedelta, timezone

def job():
    cfg=load_config(); log=setup_logger(level=cfg["env"]["LOG_LEVEL"])
    store=DataStorage(cfg["env"]["DUCKDB_PATH"])
    fc=FuturesCollector(cfg["env"]["BINANCE_API_KEY"], cfg["env"]["BINANCE_SECRET_KEY"])
    sym=cfg["data"]["symbol_internal"]; tf=cfg["data"]["timeframe"]
    last=store.last_timestamp_ms(sym, tf)
    df=fc.fetch_incremental(sym, tf, last, cfg["data"]["fetch_limit"])
    if not df.empty:
        n=store.upsert_ohlcv(sym, tf, df)
        log.info(f"UPSERT {n} rows; last={df['open_time'].max()}")

if __name__=="__main__":
    cfg=load_config(); log=setup_logger(level=cfg["env"]["LOG_LEVEL"])
    store=DataStorage(cfg["env"]["DUCKDB_PATH"])
    fc=FuturesCollector(cfg["env"]["BINANCE_API_KEY"], cfg["env"]["BINANCE_SECRET_KEY"])
    if store.last_timestamp_ms(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"]) is None:
        since=int((datetime.now(timezone.utc)-timedelta(days=cfg["data"]["backfill_days"])).timestamp()*1000)
        seed=fc.fetch_incremental(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"], since)
        if not seed.empty: store.upsert_ohlcv(cfg["data"]["symbol_internal"], cfg["data"]["timeframe"], seed)
    schedule.every(1).minutes.do(job)
    while True:
        schedule.run_pending(); time.sleep(1)
