"""Single-owner adaptive scorer. API owns updates; Kafka forwards to the API."""
import json
import os
import threading
import joblib
import pandas as pd
from river import drift
from xgboost import XGBClassifier
from src.config import MODEL_DIR

class AdaptiveFusionScorer:
    ALPHA_STABLE=.8
    ALPHA_DRIFT=.5
    ALPHA_RECOVER=.005
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir=model_dir
        self.lock=threading.RLock()
        self.xgb_model=XGBClassifier();self.xgb_model.load_model(model_dir/'xgb_model.json')
        self.xgb_model.set_params(n_jobs=1)
        self.feature_cols=joblib.load(model_dir/'feature_cols.pkl')
        self.preprocessor=joblib.load(model_dir/'preprocessor.pkl')
        self.metrics=json.loads((model_dir/'training_metrics.json').read_text())
        self.version=self.metrics['model_version']
        self.threshold=self.metrics['threshold']
        self.river_model=joblib.load(model_dir/'river_model.pkl')
        self.alpha=.8;self.adwin=drift.ADWIN();self.in_drift=False
        self.drift_count=0;self.stable_count=0;self.n_updates=0
        self.feedback_ids=set()
        path=model_dir/'online_state.pkl'
        if path.exists():
            saved=joblib.load(path)
            if saved.pop('version')==self.version: self.__dict__.update(saved)

    def predict_one(self,x_dict,y_true=None):
        if y_true is not None: raise ValueError('Use explicit feedback; prediction never trains')
        with self.lock:
            frame=pd.DataFrame([x_dict],columns=self.feature_cols)
            batch=float(self.xgb_model.predict_proba(frame)[0,1])
            online=float(self.river_model.predict_proba_one(x_dict).get(True,0.))
            score=self.alpha*batch+(1-self.alpha)*online
            return {'xgb_score':batch,'river_score':online,'fused_score':score,
                'label':int(score>=self.threshold),'alpha':self.alpha,
                'drift_active':self.in_drift,'drift_count':self.drift_count}

    def learn_feedback(self,event_id,features,label,predicted_label):
        with self.lock:
            if event_id in self.feedback_ids: return False
            self.adwin.update(int(predicted_label!=label))
            if self.adwin.drift_detected:
                self.alpha=self.ALPHA_DRIFT;self.in_drift=True
                self.drift_count+=1;self.stable_count=0
            elif self.in_drift:
                self.stable_count+=1
                self.alpha=min(self.ALPHA_STABLE,self.alpha+self.ALPHA_RECOVER)
                self.in_drift=self.alpha<self.ALPHA_STABLE
            self.river_model.learn_one(features,label)
            self.n_updates+=1;self.feedback_ids.add(event_id)
            # Atomic checkpoint includes the deduplication ID and learned state.
            self.save()
            return True

    def save(self):
        with self.lock:
            state={k:getattr(self,k) for k in ('river_model','alpha','adwin','in_drift','drift_count','stable_count','n_updates','feedback_ids')}
            state['version']=self.version
            path=self.model_dir/'online_state.pkl';tmp=path.with_suffix('.tmp')
            joblib.dump(state,tmp);os.replace(tmp,path)

    def get_status(self):
        with self.lock:
            return {'alpha':self.alpha,'in_drift':self.in_drift,'drift_count':self.drift_count,
                    'stable_count':self.stable_count,'n_updates':self.n_updates}
