import ccxt, pandas as pd, time

class FuturesCollector:
    def __init__(self, api_key="", api_secret=""):
        self.ex = ccxt.binanceusdm({
            "enableRateLimit": True,
            "apiKey": api_key or None,
            "secret": api_secret or None,
            "options": {"defaultType": "future"}  # USDT-M Futures
        })
        self.ex.load_markets()

    def fetch_ohlcv(self, symbol="BTC/USDT", timeframe="1m", since=None, limit=1000):
        o = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
        df["open_time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["open_time","open","high","low","close","volume"]]

    def fetch_incremental(self, symbol, timeframe, since_ms=None, limit=1000):
        out=[]; s=since_ms
        while True:
            df=self.fetch_ohlcv(symbol,timeframe,s,limit)
            if df.empty: break
            out.append(df)
            s=int(df["open_time"].max().timestamp()*1000)+1
            if len(df)<limit: break
            time.sleep(0.1)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
