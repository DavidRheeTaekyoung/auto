# -*- coding: utf-8 -*-
# 초간단 1분봉 전환 전략: EMA(9/21) 크로스 + 1% SL, +1% 달성 후 이익 30% 반납 시 익절
# 위치: C:\automation\BTC\simple_reversal.py
import ccxt, time
import pandas as pd
import numpy as np
import random
import os
from dataclasses import dataclass
from src.utils.config import load_config
from src.processors.data_storage import DataStorage

# =========================
# 설정
# =========================
SYMBOL       = "BTC/USDT"   # Binance USDT-M 선물
TIMEFRAME    = "1m"
BACKFILL_D   = 7            # 백테스트 기본: 과거 7일
LEV          = 10           # 레버리지 기본값 (25 → 10으로 감소)
SL_PCT       = 0.005        # 고정 손절 0.5% (1% → 0.5%로 강화)
TP_ARM_PCT   = 0.01         # 트레일링 활성화 임계(이익 1% 이상)
GIVEBACK     = 0.30         # 이익의 30% 반납 시 익절
NOTIONAL_USD = 10_000       # 모의 거래 명목가(백테/페이퍼용)
FEE_BP       = 6            # 왕복 수수료 bp 가정
SLIP_BP      = 5            # 왕복 슬리피지 bp 가정

# =========================
# 유틸
# =========================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

@dataclass
class Position:
    side: int      # 1=long, -1=short
    entry: float
    qty: float
    peak: float    # 롱: 최고가, 숏: 최저가
    armed: bool    # 트레일링 활성화 여부(이익 1% 넘은 뒤)

def fetch_ohlcv_1m_usdm(days=BACKFILL_D):
    """Binance API에서 데이터 가져오기 (현재 차단됨)"""
    try:
        ex = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType":"future"}})
        ex.load_markets()
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp()*1000)
        all_rows=[]; s=since
        while True:
            o = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=s, limit=1000)
            if not o: break
            all_rows += o
            if len(o) < 1000: break
            s = o[-1][0] + 1
            time.sleep(0.1)
        df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop_duplicates("ts").set_index("ts").sort_index()
        return df
    except Exception as e:
        print(f"❌ Binance API 오류: {e}")
        print("📊 저장된 DuckDB 데이터를 사용합니다...")
        return None

def load_from_duckdb(days=7):
    """DuckDB에서 저장된 데이터 로드"""
    try:
        cfg = load_config()
        db = DataStorage(cfg["env"]["DUCKDB_PATH"])
        
        # BTCUSDT로 조회 (슬래시 제거)
        symbol_db = "BTCUSDT"
        raw_data = db.load(symbol_db, "1m")
        
        if len(raw_data) == 0:
            print("❌ DuckDB에 데이터가 없습니다")
            return None
        
        # 최근 N일 데이터만 사용
        if days > 0:
            # 전체 데이터에서 최근 N일만 선택
            total_days = len(raw_data) / (24 * 60)  # 전체 일수
            if total_days > days:
                # 최근 N일 데이터만 선택
                start_idx = int(len(raw_data) * (1 - days / total_days))
                raw_data = raw_data.iloc[start_idx:]
        
        # DataFrame 형태로 변환
        df = raw_data.copy()
        df = df.set_index(pd.to_datetime(df['open_time']))
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ DuckDB에서 데이터 로드 완료: {len(df):,}행")
        print(f"기간: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"❌ DuckDB 로드 실패: {e}")
        return None

def signal_engine(df: pd.DataFrame):
    """EMA9-EMA21 크로스 + 가격 0.3% 이상 움직임 확인"""
    x=df.copy()
    x["ema9"]=ema(x["close"],9)
    x["ema21"]=ema(x["close"],21)
    x["cr"] = np.sign(x["ema9"] - x["ema21"])
    x["cr_prev"] = x["cr"].shift(1)
    
    # 가격 변화율 계산 (이전 5분 대비)
    x["price_change"] = x["close"].pct_change(5)
    
    # 방향 전환 + 가격 0.3% 이상 움직임 확인
    # +1: 골든(롱 진입) + 가격 0.3% 이상 상승
    # -1: 데드(숏 진입) + 가격 0.3% 이상 하락
    sig = np.where(
        (x["cr_prev"]<=0) & (x["cr"]>0) & (x["price_change"] > 0.003), 1,
        np.where(
            (x["cr_prev"]>=0) & (x["cr"]<0) & (x["price_change"] < -0.003), -1, 0
        )
    )
    x["sig"]=sig
    return x[["open","high","low","close","volume","ema9","ema21","price_change","sig"]].dropna()

def apply_trailing(pos: Position, price_now: float):
    """+1% 달성 시 트레일링 활성화 → 이익의 30% 반납 시 청산"""
    if pos.side==1:
        # 이익률
        gain = (price_now/pos.entry) - 1.0
        if not pos.armed and gain >= TP_ARM_PCT:
            pos.armed = True
            pos.peak = max(pos.peak, price_now)
        if pos.armed:
            pos.peak = max(pos.peak, price_now)
            # 이익의 70% 보전(=30% 반납 시점에서 청산)
            trigger = pos.entry * (1 + (pos.peak/pos.entry - 1)*(1 - GIVEBACK))
            if price_now <= trigger:  # 트레일 히트
                return True
        return False
    else:
        gain = (pos.entry/price_now) - 1.0
        if not pos.armed and gain >= TP_ARM_PCT:
            pos.armed = True
            pos.peak = min(pos.peak, price_now)
        if pos.armed:
            pos.peak = min(pos.peak, price_now)
            trigger = pos.entry * (1 - (1 - pos.peak/pos.entry)*(1 - GIVEBACK))
            if price_now >= trigger:
                return True
        return False

def stop_hit(pos: Position, price_now: float):
    if pos.side==1:  # 롱: -1% 손절
        return price_now <= pos.entry*(1 - SL_PCT)
    else:            # 숏: -1% 손절
        return price_now >= pos.entry*(1 + SL_PCT)

def backtest(df: pd.DataFrame, notional_usd=NOTIONAL_USD, fee_bp=FEE_BP, slip_bp=SLIP_BP):
    x = signal_engine(df)
    eq=10000.0
    equity=[]; trades=[]
    pos: Position|None = None
    for t, row in x.iterrows():
        px = float(row["close"])
        sig = int(row["sig"])
        # 진입
        if pos is None and sig!=0:
            qty = (notional_usd*LEV)/px
            pos = Position(side=sig, entry=px, qty=qty, peak=px, armed=False)
            # 즉시 비용(반 왕복 반영 보수적으로)
            eq -= (qty*px)*(fee_bp/2/1e4 + slip_bp/2/1e4)
        # 보유 중 종료 조건
        if pos is not None:
            if stop_hit(pos, px) or apply_trailing(pos, px):
                pnl = pos.qty*(px - pos.entry)*pos.side
                cost = (pos.qty*px)*(fee_bp/2/1e4 + slip_bp/2/1e4)
                eq += pnl - cost
                trades.append({"time":t, "side":"LONG" if pos.side==1 else "SHORT",
                               "entry":pos.entry, "exit":px, "pnl":pnl, "eq":eq})
                pos=None
        equity.append(eq if pos is None else eq + pos.qty*(px-pos.entry)*pos.side)
    res=pd.DataFrame({"eq":equity}, index=x.index)
    # 성과
    ret = res["eq"].pct_change().fillna(0)
    sharpe = (ret.mean()/(ret.std()+1e-12))*np.sqrt(365*24*60)  # 1분 기준
    dd = (res["eq"]/res["eq"].cummax()-1).min()
    out = {
        "final_eq": float(res["eq"].iloc[-1]),
        "sharpe": float(sharpe),
        "max_dd": float(dd),
        "trades": pd.DataFrame(trades)
    }
    return out

def random_period_backtest(df: pd.DataFrame, num_periods=3, period_days=3):
    """랜덤 기간 3개를 뽑아서 백테스트"""
    print(f"\n🎯 랜덤 기간 {num_periods}개 백테스트 시작")
    print(f"📅 각 기간: {period_days}일")
    
    # 전체 데이터 길이
    total_rows = len(df)
    period_minutes = period_days * 24 * 60  # 3일 = 4320분
    
    # 랜덤 시드 설정 (재현 가능하도록)
    random.seed(42)
    
    results = []
    
    for i in range(num_periods):
        # 랜덤 시작점 선택 (마지막 기간이 끝나도록)
        max_start = total_rows - period_minutes
        if max_start <= 0:
            print(f"❌ 데이터가 부족합니다: 전체 {total_rows}행, 필요 {period_minutes}행")
            break
            
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + period_minutes
        
        # 해당 기간 데이터 추출
        period_data = df.iloc[start_idx:end_idx].copy()
        start_time = period_data.index[0]
        end_time = period_data.index[-1]
        
        print(f"\n--- 기간 {i+1} 백테스트 ---")
        print(f"시작: {start_time}")
        print(f"종료: {end_time}")
        print(f"데이터: {len(period_data):,}행")
        
        # 백테스트 실행
        try:
            bt_result = backtest(period_data)
            
            # 결과 저장
            period_result = {
                'period': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'final_eq': bt_result['final_eq'],
                'sharpe': bt_result['sharpe'],
                'max_dd': bt_result['max_dd'],
                'trades_count': len(bt_result['trades']) if len(bt_result['trades']) > 0 else 0,
                'return_pct': ((bt_result['final_eq'] - 10000) / 10000) * 100
            }
            
            results.append(period_result)
            
            print(f"✅ 완료: 수익률 {period_result['return_pct']:+.2f}%, Sharpe {period_result['sharpe']:.2f}")
            
        except Exception as e:
            print(f"❌ 기간 {i+1} 백테스트 실패: {e}")
            continue
    
    # 전체 결과 종합
    if results:
        print("\n" + "="*60)
        print("🏆 랜덤 기간 백테스트 결과 종합")
        print("="*60)
        
        for result in results:
            print(f"기간 {result['period']}: {result['start_time'].strftime('%Y-%m-%d %H:%M')} ~ {result['end_time'].strftime('%Y-%m-%d %H:%M')}")
            print(f"  수익률: {result['return_pct']:+.2f}%")
            print(f"  Sharpe: {result['sharpe']:6.2f}")
            print(f"  Max DD: {result['max_dd']*100:+.1f}%")
            print(f"  거래 수: {result['trades_count']}건")
            print()
        
        # 평균 성과
        avg_return = np.mean([r['return_pct'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_dd = np.mean([r['max_dd'] for r in results])
        total_trades = sum([r['trades_count'] for r in results])
        
        print(f"📊 전체 평균:")
        print(f"  평균 수익률: {avg_return:+.2f}%")
        print(f"  평균 Sharpe: {avg_sharpe:.2f}")
        print(f"  평균 Max DD: {avg_dd*100:+.1f}%")
        print(f"  총 거래 수: {total_trades}건")
        
        # 승률 계산
        winning_periods = len([r for r in results if r['return_pct'] > 0])
        win_rate = (winning_periods / len(results)) * 100
        print(f"  승률: {winning_periods}/{len(results)} ({win_rate:.1f}%)")
        
        return results
    else:
        print("❌ 모든 기간에서 백테스트 실패")
        return []

# =========================
# 실행: 백테스트 → (옵션) 페이퍼 라이브 루프
# =========================
if __name__=="__main__":
    print("[1/3] 데이터 로드(선물 1분봉, 7일)…")
    
    # 먼저 Binance API 시도, 실패하면 DuckDB 사용
    df = fetch_ohlcv_1m_usdm(BACKFILL_D)
    if df is None:
        df = load_from_duckdb(BACKFILL_D)
        if df is None:
            print("❌ 데이터를 로드할 수 없습니다. 프로그램을 종료합니다.")
            exit(1)
    
    print(f"데이터 로드 완료: {len(df):,}행")
    print(f"기간: {df.index[0]} ~ {df.index[-1]}")
    print(df.tail())

    print("\n[2/3] 전체 기간 백테스트 실행…")
    bt = backtest(df)
    print({k: (round(v,4) if isinstance(v, float) else v.shape) for k,v in bt.items() if k!="trades"})
    print(bt["trades"].tail(10))

    print("\n[3/3] 랜덤 기간 3개 백테스트 실행…")
    random_results = random_period_backtest(df, num_periods=3, period_days=3)

    # ---- 페이퍼 라이브 (원하면 주석 해제) ----
    # ex = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType":"future"}})
    # pos=None; eq=10000.0
    # while True:
    #     tkr=ex.fetch_ticker(SYMBOL)
    #     px=(tkr["bid"]+tkr["ask"])/2
    #     # 간단 데모: 직전 100개 캔들로 시그널 재계산 후 동일 로직 적용
    #     ohlcv=ex.fetch_ohlcv(SYMBOL,TIMEFRAME,limit=150)
    #     df_live=pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    #     df_live["ts"]=pd.to_datetime(df_live["ts"],unit="ms",utc=True)
    #     df_live["ts"]=df_live.set_index("ts")
    #     # 여기서 signal_engine/stop_hit/apply_trailing 재사용 → 주문 연동은 추후 broker 클래스 연결
    #     print(px)
    #     time.sleep(60)
