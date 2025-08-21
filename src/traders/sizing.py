import numpy as np

def confidence_from_quantiles(q10,q50,q90):
    width = max(q90 - q10, 1e-6)
    c = min(abs(q50)/width, 1.0)           # 0~1
    direction = 1 if q50>0 else -1
    return c, direction

def leverage_from_confidence(c, max_leverage=25, base=5):
    # 확신도 0→base배, 1→max배
    return float(min(max_leverage, base + (max_leverage-base)*c))

def kelly_fraction(win_prob, payoff_ratio, base=0.10):
    # 분수 Kelly (기본 base 가중), payoff_ratio = avg_win/avg_loss
    p=win_prob; q=1-p; b=max(payoff_ratio,1e-6)
    raw=(p*b - q)/b
    return float(max(0.0, base*max(0.0, raw)))

def position_size_usd(cash_usdt, price, lev, cap_usd):
    # 겉보기 명목가 = cash*lev, 상한 cap 적용
    notional = min(cash_usdt*lev, cap_usd)
    qty = notional / price                 # 선물(USDT-M) 선형
    return notional, qty
