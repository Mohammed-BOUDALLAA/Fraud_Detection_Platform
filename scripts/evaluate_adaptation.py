"""Separate prequential experiment. Immediate labels are an optimistic assumption."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from river import drift
from xgboost import XGBClassifier
from src import config
from src.models.train import metrics

def main():
    manifest=json.loads((config.PROC_DIR/'dataset_manifest.json').read_text())
    metadata=json.loads((config.MODEL_DIR/'training_metrics.json').read_text())
    data=pd.read_parquet(config.PROC_DIR/'features.parquet').iloc[manifest['validation_end']:]
    cols=joblib.load(config.MODEL_DIR/'feature_cols.pkl')
    model=XGBClassifier();model.load_model(config.MODEL_DIR/'xgb_model.json');model.set_params(n_jobs=1)
    frozen=joblib.load(config.MODEL_DIR/'river_model.pkl');online=copy.deepcopy(frozen)
    batch=model.predict_proba(data[cols])[:,1]
    fixed=[];adaptive=[];trace=[];alpha=.8;detector=drift.ADWIN();alarms=0
    for row,label,batch_score in zip(data[cols].to_dict('records'),data.isFraud,batch):
        fixed.append(.8*batch_score+.2*frozen.predict_proba_one(row).get(True,0.))
        score=alpha*batch_score+(1-alpha)*online.predict_proba_one(row).get(True,0.)
        adaptive.append(score);trace.append(alpha)
        detector.update(int((score>=metadata['threshold'])!=label))
        if detector.drift_detected:alpha=.5;alarms+=1
        else:alpha=min(.8,alpha+.005)
        online.learn_one(row,int(label))
    report={'dataset':manifest,'assumption':'verified labels arrive immediately after each prediction; optimistic replay',
        'threshold':metadata['threshold'],'fixed_fusion':metrics(data.isFraud,np.array(fixed),metadata['threshold']),
        'adaptive_fusion':metrics(data.isFraud,np.array(adaptive),metadata['threshold']),
        'drift_count':alarms,'alpha_trace':trace}
    (config.MODEL_DIR/'adaptation_evaluation.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='alpha_trace'})
if __name__=='__main__':main()
