import duckdb, pandas as pd, os
from datetime import timezone

class DataStorage:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv(
              symbol VARCHAR,
              timeframe VARCHAR,
              open_time TIMESTAMP,
              open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
              volume DOUBLE
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx ON ohlcv(symbol,timeframe,open_time)")

    def upsert_ohlcv(self, symbol, timeframe, df: pd.DataFrame):
        if df.empty: return 0
        # 컬럼 순서를 테이블 스키마에 맞춤
        t = df.copy()
        t = t[["open_time", "open", "high", "low", "close", "volume"]]
        t["symbol"] = symbol
        t["timeframe"] = timeframe
        # 테이블 스키마 순서로 컬럼 재배열
        t = t[["symbol", "timeframe", "open_time", "open", "high", "low", "close", "volume"]]
        
        self.conn.register("tmpdf", t)
        self.conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS new_rows AS SELECT * FROM tmpdf;
            DELETE FROM ohlcv using new_rows n
            WHERE ohlcv.symbol=n.symbol AND ohlcv.timeframe=n.timeframe AND ohlcv.open_time=n.open_time;
            INSERT INTO ohlcv SELECT * FROM new_rows;
        """)
        return len(t)

    def last_timestamp_ms(self, symbol, timeframe):
        r=self.conn.execute("SELECT MAX(open_time) FROM ohlcv WHERE symbol=? AND timeframe=?",
                            [symbol,timeframe]).fetchone()
        if not r or not r[0]: return None
        return int(r[0].replace(tzinfo=timezone.utc).timestamp()*1000)

    def load(self, symbol, timeframe):
        return self.conn.execute("SELECT * FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY open_time",
                                 [symbol,timeframe]).df()
