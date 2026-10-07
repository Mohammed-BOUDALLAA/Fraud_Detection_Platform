"""Authenticated read-only benchmark. Unique IDs avoid idempotency replay timing."""
import os
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import requests
from src import config
token=os.environ['FRAUD_API_TOKEN']
cols=joblib.load(config.MODEL_DIR/'feature_cols.pkl')
rows=pd.read_parquet(config.PROC_DIR/'features.parquet').tail(100)[cols]
times=[]
with requests.Session() as session:
    for row in rows.to_dict('records'):
        t=time.perf_counter()
        r=session.post(config.API_URL+'/predict',headers={'Authorization':'Bearer '+token},
            json={'transaction_id':str(uuid.uuid4()),'features':row},timeout=30)
        r.raise_for_status();times.append((time.perf_counter()-t)*1000)
print(dict(zip(['p50_ms','p95_ms','p99_ms'],np.percentile(times,[50,95,99]).tolist())))
