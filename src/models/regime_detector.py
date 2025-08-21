import numpy as np, pandas as pd
from hmmlearn import hmm

class Regime:
    names=['bull','bear','sideways','high_vol','low_vol']

class RegimeDetector:
    def __init__(self, n_states=5, seed=42):
        self.m=hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed)
        self.n=n_states; self.fitted=False

    def _feat(self, df):
        r=np.log(df["close"]/df["close"].shift(1)).fillna(0).values.reshape(-1,1)
        v=pd.Series(r.ravel()).rolling(60).std().bfill().values.reshape(-1,1)
        z=((df["volume"]-df["volume"].rolling(100).mean())/(df["volume"].rolling(100).std()+1e-8)).fillna(0).values.reshape(-1,1)
        return np.hstack([r,v,z])

    def fit(self, df):
        X=self._feat(df); self.m.fit(X); self.fitted=True

    def current(self, df):
        if not self.fitted: return "unknown", np.zeros(self.n)
        X=self._feat(df)
        s=self.m.predict(X)[-1]; p=self.m.predict_proba(X)[-1]
        return Regime.names[s], p
