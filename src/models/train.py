"""Train XGBoost + River on train only; choose threshold on validation; test once."""
import argparse
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
from river import preprocessing, linear_model
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from src.config import PROC_DIR, MODEL_DIR

def metrics(y,p,threshold):
    pred=p>=threshold
    return {'auc_pr':float(average_precision_score(y,p)), 'auc_roc':float(roc_auc_score(y,p)),
            'f1_fraud':float(f1_score(y,pred,zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def train(n_estimators=300, online_rows=50000):
    df=pd.read_parquet(PROC_DIR/'features.parquet')
    manifest=json.loads((PROC_DIR/'dataset_manifest.json').read_text())
    cols=joblib.load(MODEL_DIR/'feature_cols.pkl')
    X,y=df[cols],df.isFraud
    a,b=manifest['train_end'],manifest['validation_end']
    for label,part in [('train',y[:a]),('validation',y[a:b]),('test',y[b:])]:
        if part.nunique()!=2: raise ValueError(f'{label} needs both classes for evaluation')
    model=XGBClassifier(n_estimators=n_estimators,max_depth=5,learning_rate=.05,
        subsample=.8,colsample_bytree=.8,tree_method='hist',n_jobs=2,random_state=42,
        scale_pos_weight=float((y[:a]==0).sum()/y[:a].sum()),eval_metric='aucpr',early_stopping_rounds=20)
    model.fit(X[:a],y[:a],eval_set=[(X[a:b],y[a:b])],verbose=False)
    online=preprocessing.StandardScaler() | linear_model.LogisticRegression()
    start=max(0,a-online_rows)
    for row,target in zip(X.iloc[start:a].to_dict('records'),y.iloc[start:a]): online.learn_one(row,int(target))
    def scores(part):
        batch=model.predict_proba(part)[:,1]
        stream=np.array([online.predict_proba_one(row).get(True,0.) for row in part.to_dict('records')])
        return batch,stream,.8*batch+.2*stream
    _,_,valid=scores(X[a:b])
    grid=np.linspace(.01,.99,99)
    threshold=float(max(grid,key=lambda t:f1_score(y[a:b],valid>=t,zero_division=0)))
    bp,rp,fp=scores(X[b:])
    model.save_model(MODEL_DIR/'xgb_model.json')
    joblib.dump(model,MODEL_DIR/'xgb_model.pkl')
    joblib.dump(online,MODEL_DIR/'river_model.pkl')
    version=hashlib.sha256((MODEL_DIR/'xgb_model.json').read_bytes()+(MODEL_DIR/'preprocessor.pkl').read_bytes()+(MODEL_DIR/'river_model.pkl').read_bytes()+str(threshold).encode()).hexdigest()[:16]
    report={**metrics(y[b:],fp,threshold),'threshold':threshold,'model_version':version,
        'dataset':manifest,'batch_test':metrics(y[b:],bp,.5),'river_test':metrics(y[b:],rp,.5),
        'evaluation':'frozen fusion on held-out temporal test; adaptive benefit not established',
        'online_warmup_rows':a-start,'best_iteration':int(model.best_iteration),'python':platform.python_version()}
    (MODEL_DIR/'training_metrics.json').write_text(json.dumps(report,indent=2))
    (MODEL_DIR/'online_state.pkl').unlink(missing_ok=True)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trees',type=int,default=300)
    parser.add_argument('--online-rows',type=int,default=50000)
    args=parser.parse_args();train(args.trees,args.online_rows)
