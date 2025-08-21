import ccxt, time

class BinanceUSDM:
    def __init__(self, api_key, api_secret, mode="SIM"):
        self.mode=mode
        self.ex = ccxt.binanceusdm({
            "enableRateLimit": True,
            "apiKey": api_key or None,
            "secret": api_secret or None,
            "options": {"defaultType":"future"}
        })
        self.symbol="BTC/USDT"

    # ===== 계좌/포지션 조회 (LIVE에서만 실제 호출) =====
    def balance(self):
        if self.mode=="SIM":
            return {"USDT":{"free":1_000_000}}
        b=self.ex.fetch_balance()
        return b["USDT"]

    def position_qty(self):
        if self.mode=="SIM": return 0.0
        pos=self.ex.fetch_positions_risk([self.symbol])[0]
        return float(pos["contracts"]) if pos else 0.0

    # ===== 주문 =====
    def market_open(self, side, qty):
        if self.mode=="SIM": return {"id":"SIM-OPEN", "price": self.mid()}
        return self.ex.create_order(self.symbol,"MARKET", side, qty, None, {"reduceOnly": False})

    def stop_close(self, side, stop_price, qty):
        # side: LONG 청산이면 'SELL', SHORT 청산이면 'BUY'
        params={"reduceOnly": True, "stopPrice": float(stop_price)}
        if self.mode=="SIM": return {"id":"SIM-STOP", "stop":stop_price}
        # Binance Futures STOP_MARKET
        return self.ex.create_order(self.symbol,"STOP_MARKET", side, qty, None, params)

    def cancel_all(self):
        if self.mode=="SIM": return
        try: self.ex.cancel_all_orders(self.symbol)
        except Exception: pass

    def mid(self):
        t=self.ex.fetch_ticker(self.symbol); return float((t["bid"]+t["ask"])/2)
