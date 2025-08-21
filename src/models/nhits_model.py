import torch, torch.nn as nn, numpy as np

class Pinball(nn.Module):
    def __init__(self,q): super().__init__(); self.q=q
    def forward(self, yhat,y):
        e=y-yhat
        return torch.max((self.q-1)*e, self.q*e).mean()

class NHiTSLite(nn.Module):
    def __init__(self, in_feat, horizons=(60,240,1440), quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.h= horizons; self.qs=quantiles
        self.pool=nn.AdaptiveAvgPool1d(64)
        self.mlp=nn.Sequential(
            nn.Linear(64*in_feat,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256,len(horizons)*len(quantiles))
        )
    def forward(self,x):          # x:[B,T,F]
        x=x.transpose(1,2)        # [B,F,T]
        x=self.pool(x).reshape(x.size(0),-1)
        out=self.mlp(x)
        return out.view(x.size(0), len(self.h), len(self.qs))

class Predictor:
    def __init__(self, in_feat, horizons, quantiles, lr=7e-4, device=None):
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model=NHiTSLite(in_feat,horizons,quantiles).to(self.device)
        self.qs=quantiles; self.opt=torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.losses=[Pinball(q) for q in self.qs]
    def fit(self,X,Y,epochs=20,batch=256):
        ds=torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
        dl=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=True,drop_last=True)
        for _ in range(epochs):
            self.model.train()
            for xb,yb in dl:
                xb,yb=xb.to(self.device), yb.to(self.device)
                pred=self.model(xb)
                loss=sum(self.losses[i](pred[:,:,i], yb) for i in range(len(self.qs)))
                self.opt.zero_grad(); loss.backward(); self.opt.step()
    def predict(self, X1):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X1,device=self.device)).cpu().numpy()[0]  # [H,Q]
