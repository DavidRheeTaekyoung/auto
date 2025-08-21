import numpy as np, pandas as pd

def enrich(df: pd.DataFrame)->pd.DataFrame:
    x=df.copy()
    x["ret1"]=np.log(x["close"]/x["close"].shift(1)).fillna(0.0)
    x["vol20"]=x["ret1"].rolling(20).std().bfill()
    x["vol60"]=x["ret1"].rolling(60).std().bfill()
    d=x["close"].diff()
    up=d.clip(lower=0).rolling(14).mean()
    dn=(-d.clip(upper=0)).rolling(14).mean()
    rs=(up/(dn+1e-8)).replace([np.inf,-np.inf],0)
    x["rsi14"]=100-100/(1+rs)
    ma20=x["close"].rolling(20).mean(); sd20=x["close"].rolling(20).std()
    x["bb_up"]=(ma20+2*sd20).bfill(); x["bb_dn"]=(ma20-2*sd20).bfill()
    x["vz"]=(x["volume"]-x["volume"].rolling(100).mean())/(x["volume"].rolling(100).std()+1e-8)
    return x.dropna().reset_index(drop=True)

def make_dataset(df: pd.DataFrame, input_window=512, horizons=(60,240,1440)):
    F=["open","high","low","close","volume","ret1","vol20","vol60","rsi14","bb_up","bb_dn","vz"]
    A=df[F].values; C=df["close"].values
    X=[]; Y=[]
    for i in range(input_window, len(df)-max(horizons)):
        X.append(A[i-input_window:i])
        p0=C[i-1]; Y.append([(C[i+h-1]-p0)/p0 for h in horizons])
    return (np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32))
