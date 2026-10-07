"""Explicit synthetic integration fixture. Never use these metrics as IEEE-CIS results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.config import DATA_DIR
from src.models.feature_engineering import build
rng=np.random.default_rng(42)
n=3000
raw=DATA_DIR/'demo_raw';raw.mkdir(exist_ok=True)
amount=rng.lognormal(4,1.2,n)
card=rng.integers(1000,1200,n)
product=rng.choice(['W','C','R','H'],n)
p=1/(1+np.exp(-(-4+1.6*(amount>150)+1.5*(product=='C'))))
pd.DataFrame({'TransactionID':np.arange(n)+100000,'TransactionDT':np.arange(n)*300,
 'TransactionAmt':amount,'card1':card,'card2':rng.integers(100,600,n),
 'ProductCD':product,'P_emaildomain':rng.choice(['gmail.com','yahoo.com','protonmail.com'],n),
 'R_emaildomain':rng.choice(['gmail.com','hotmail.com'],n),
 'V1':rng.normal(size=n),'V2':rng.normal(size=n), 'isFraud':rng.binomial(1,p)}).to_csv(raw/'train_transaction.csv',index=False)
pd.DataFrame({'TransactionID':np.arange(n)[::2]+100000,'DeviceType':rng.choice(['mobile','desktop'],n//2)}).to_csv(raw/'train_identity.csv',index=False)
build(raw,'SYNTHETIC integration fixture, seed 42; NOT IEEE-CIS')
