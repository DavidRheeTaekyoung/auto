from src.collectors.ohlcv_collector import FuturesCollector
from src.processors.data_storage import DataStorage
from src.features.feature_maker import enrich, make_dataset

def test_end_to_end():
    fc=FuturesCollector()
    df=fc.fetch_ohlcv("BTC/USDT","1m",limit=50)
    assert len(df)>0
    store=DataStorage("C:/automation/BTC/data/ohlcv.duckdb")
    n=store.upsert_ohlcv("BTC/USDT","1m",df); assert n>0
    raw=store.load("BTC/USDT","1m")
    feat=enrich(raw); X,Y=make_dataset(feat, input_window=32, horizons=(10,20,30))
    assert X.shape[0]>0 and X.shape[2]>0
