import numpy as np
class Hedge:
    def __init__(self,n,eta=0.05): self.w=np.ones(n)/n; self.c=np.zeros(n); self.eta=eta
    def combine(self, preds):      # preds: [n_models,H,Q]
        w=self.w/(self.w.sum()+1e-8)
        return np.tensordot(w, preds, axes=(0,0))
    def update(self, losses):      # 각 모델 손실
        self.c+=losses; self.w=np.exp(-self.eta*self.c); self.w/=self.w.sum()
