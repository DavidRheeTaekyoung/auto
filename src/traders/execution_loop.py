import numpy as np, time
from src.traders.sizing import confidence_from_quantiles, leverage_from_confidence, kelly_fraction, position_size_usd
from src.traders.risk import stop_price_for_account_risk, TrailingManager

class TradeEngine:
    def __init__(self, broker, cfg):
        self.broker=broker; self.cfg=cfg; self.active=None
        self.trailer=None

    def decide_and_execute(self, price, q10,q50,q90, cash_usdt):
        c, dirn = confidence_from_quantiles(q10,q50,q90)
        lev      = leverage_from_confidence(c, self.cfg["trade"]["max_leverage"], base=5)
        # 확률 근사: q50>0이면 롱승률↑, 반대면 숏승률↑ (간단화)
        win_prob = 0.5 + 0.4*c
        payoff   = max(abs(q90/q10), 1.0) if q10!=0 else 1.5
        kfrac    = kelly_fraction(win_prob, payoff, base=self.cfg["trade"]["kelly_base"])
        eff_lev  = min(lev, self.cfg["trade"]["max_leverage"]*kfrac*2 + 1)  # 과격 방지 완충

        notional, qty = position_size_usd(cash_usdt, price, eff_lev, self.cfg["trade"]["max_position_usd"])
        side = "BUY" if dirn>0 else "SELL"

        # 엔트리
        order = self.broker.market_open(side, qty)
        entry = price if "price" not in order else order["price"]

        # 손절가 (계좌 10% 손실 기준)
        stop = stop_price_for_account_risk(entry, qty, cash_usdt, is_long=(side=="BUY"),
                                           cash_risk_pct=self.cfg["trade"]["stoploss_cash_pct"])

        # 트레일링 매니저
        self.trailer = TrailingManager(self.cfg["trade"]["trailing_giveback_pct"])

        # 스탑 주문 생성(감소 전용)
        close_side = "SELL" if side=="BUY" else "BUY"
        self.broker.cancel_all()
        self.broker.stop_close(close_side, stop, qty)

        self.active={"side":side,"qty":qty,"entry":entry,"stop":stop,"close_side":close_side}
        return {"entry":entry,"qty":qty,"side":side,"lev":eff_lev,"c":c}

    def on_price_tick(self, last_price):
        if not self.active: return None
        # 미실현 PnL
        s=1 if self.active["side"]=="BUY" else -1
        pnl = self.active["qty"] * (last_price - self.active["entry"]) * s
        self.trailer.update(max(pnl,0.0))
        # 트레일링 스탑 재설정
        trail_price = self.trailer.trailing_stop_price(self.active["entry"], self.active["qty"], is_long=(s==1))
        # 스탑이 진입가를 넘지 않게(롱은 위로만, 숏은 아래로만 조정)
        if (s==1 and trail_price>self.active["stop"]) or (s==-1 and trail_price<self.active["stop"]):
            self.active["stop"]=trail_price
            self.broker.cancel_all()
            self.broker.stop_close(self.active["close_side"], self.active["stop"], self.active["qty"])
        return {"pnl":pnl,"stop":self.active["stop"]}
