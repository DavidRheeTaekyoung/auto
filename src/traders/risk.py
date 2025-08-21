def stop_price_for_account_risk(entry_price, qty, cash_usdt, is_long, cash_risk_pct=0.10):
    risk = cash_usdt * cash_risk_pct
    # PnL = qty * (P_exit - entry) [USDT-M 선형]
    if is_long:
        stop = entry_price - risk/qty
    else:
        stop = entry_price + risk/qty
    return max(1.0, stop)

class TrailingManager:
    """
    목표: 실행 중 최대 이익의 70%는 보존(= 30% giveback에서 청산)
    구현: 피크 PnL을 추적, 해당 피크의 70% 수준을 보존하는 가격으로 스탑 재조정.
    """
    def __init__(self, giveback=0.30): self.giveback=giveback; self.peak_pnl=0.0
    def update(self, unrealized_pnl):
        self.peak_pnl = max(self.peak_pnl, unrealized_pnl)
    def trailing_threshold_pnl(self):
        return self.peak_pnl*(1.0 - self.giveback)  # 이익의 70%를 보전
    def trailing_stop_price(self, entry_price, qty, is_long):
        pnl_floor = self.trailing_threshold_pnl()
        # 청산가격: entry ± pnl_floor/qty
        if is_long:
            return entry_price + pnl_floor/qty
        else:
            return entry_price - pnl_floor/qty
