from src.processors.data_storage import DataStorage

store = DataStorage('C:/automation/BTC/data/ohlcv.duckdb')

print("=== 데이터베이스 상세 상태 ===")
print("전체 데이터 수:", store.conn.execute('SELECT COUNT(*) FROM ohlcv').fetchone()[0])
print()

print("타임프레임별 데이터 수:")
result = store.conn.execute('SELECT timeframe, COUNT(*) FROM ohlcv GROUP BY timeframe').fetchall()
for tf, cnt in result:
    print(f"  {tf}: {cnt:,}행")

print()
print("1분봉 데이터 샘플:")
sample = store.conn.execute("SELECT * FROM ohlcv WHERE timeframe='1m' ORDER BY open_time DESC LIMIT 3").fetchall()
for row in sample:
    print(f"  {row}")

print()
print("1일봉 데이터 샘플:")
sample = store.conn.execute("SELECT * FROM ohlcv WHERE timeframe='1d' ORDER BY open_time DESC LIMIT 3").fetchall()
for row in sample:
    print(f"  {row}")
