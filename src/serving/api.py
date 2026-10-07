"""Local single-worker PFE API. Authentication required for prediction and feedback."""
import json
import math
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from sqlalchemy import Column, String, Text
from src import config
from src.auth.database import Base, engine, init_db, SessionLocal
from src.auth.auth_router import router
from src.auth.dependencies import get_current_user, require_admin
from src.models.fusion import AdaptiveFusionScorer
from src.serving.cache import ShapCache
from src.users.crud import log_audit

class PredictionRecord(Base):
    __tablename__='predictions'
    event_id=Column(String,primary_key=True)
    payload_hash=Column(String,nullable=False)
    model_version=Column(String,nullable=False)
    features=Column(Text,nullable=False)
    response=Column(Text,nullable=False)

@asynccontextmanager
async def lifespan(app):
    init_db()
    app.state.scorer=AdaptiveFusionScorer()
    app.state.cache=ShapCache(app.state.scorer.version)
    yield
    app.state.scorer.save()

app=FastAPI(title='Adaptive Fraud Detection — PFE',version='2.0.0',lifespan=lifespan)
app.include_router(router)

class TransactionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    transaction_id:str|None=Field(default=None,max_length=128)
    features:dict[str,float|str|None]
    input_type:Literal['processed','raw']='processed'
    ground_truth:StrictInt|None=None

class FeedbackRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    transaction_id:str
    ground_truth:StrictInt=Field(ge=0,le=1)

def prepare(request):
    scorer=app.state.scorer
    if request.ground_truth is not None:
        raise HTTPException(422,'Send verified labels to /feedback after prediction, never to /predict')
    if 'isFraud' in request.features: raise HTTPException(422,'Target leakage: remove isFraud from features')
    try:
        if request.input_type=='raw':
            for field in ('TransactionAmt','TransactionDT','card1'):
                if field not in request.features or request.features[field] is None: raise ValueError(f'Missing {field}')
            frame=scorer.preprocessor.transform(pd.DataFrame([request.features]))
            features=frame.iloc[0].to_dict()
        else:
            missing=set(scorer.feature_cols)-set(request.features)
            extra=set(request.features)-set(scorer.feature_cols)
            if missing or extra: raise ValueError(f'Feature schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}')
            features={k:float(request.features[k]) if request.features[k] is not None else -999. for k in scorer.feature_cols}
        if not all(math.isfinite(v) for v in features.values()): raise ValueError('Features must be finite')
        return features
    except (ValueError,TypeError,KeyError) as exc: raise HTTPException(422,str(exc))

def explain(features):
    scorer=app.state.scorer;cache=app.state.cache
    cached=cache.get(features)
    if cached is not None:return cached
    matrix=xgb.DMatrix(pd.DataFrame([features],columns=scorer.feature_cols))
    # Exact TreeSHAP contributions in batch model log-odds, not fused probability.
    vals=scorer.xgb_model.get_booster().predict(matrix,pred_contribs=True,
        iteration_range=(0,scorer.xgb_model.best_iteration+1))[0,:-1]
    indices=np.argsort(np.abs(vals))[-3:][::-1]
    result=[{'feature':scorer.feature_cols[i],'value':features[scorer.feature_cols[i]],'shap_value':float(vals[i])} for i in indices]
    cache.set(features,result);return result

def score_prepared(request,features):
    import hashlib
    started=time.perf_counter();scorer=app.state.scorer
    event_id=request.transaction_id or str(uuid.uuid4())
    digest=hashlib.sha256(json.dumps(features,sort_keys=True).encode()).hexdigest()
    with scorer.lock, SessionLocal() as db:
        existing=db.get(PredictionRecord,event_id)
        if existing:
            if existing.payload_hash!=digest or existing.model_version!=scorer.version:
                raise HTTPException(409,'Transaction ID already used for different input/model')
            return json.loads(existing.response)
        result=scorer.predict_one(features)
        result.update(transaction_id=event_id,shap_top3=explain(features),model_version=scorer.version,
            threshold=scorer.threshold,explanation_scope='XGBoost log-odds contributions only',
            synthetic=scorer.metrics['dataset']['synthetic'],latency_ms=(time.perf_counter()-started)*1000)
        db.add(PredictionRecord(event_id=event_id,payload_hash=digest,model_version=scorer.version,
            features=json.dumps(features),response=json.dumps(result)));db.commit()
        return result

@app.post('/predict')
def predict(request:TransactionRequest,user=Depends(get_current_user)):
    return score_prepared(request,prepare(request))

@app.post('/predict/batch')
def batch(requests:list[TransactionRequest],user=Depends(get_current_user)):
    if not 1<=len(requests)<=500:raise HTTPException(422,'Batch size must be 1–500')
    prepared=[prepare(r) for r in requests]
    return [score_prepared(r,f) for r,f in zip(requests,prepared)]

@app.post('/feedback')
def feedback(request:FeedbackRequest,user=Depends(require_admin)):
    scorer=app.state.scorer
    with scorer.lock, SessionLocal() as db:
        record=db.get(PredictionRecord,request.transaction_id)
        if record is None:raise HTTPException(404,'Predict this transaction first')
        if record.model_version!=scorer.version:raise HTTPException(409,'Feedback belongs to an older model')
        updated=scorer.learn_feedback(request.transaction_id,json.loads(record.features),
            request.ground_truth,json.loads(record.response)['label'])
        if updated:log_audit(db,user.username,'feedback',request.transaction_id)
    return {'updated':updated,**scorer.get_status()}

@app.get('/health')
def health():
    scorer=app.state.scorer
    return {'status':'healthy','synthetic':scorer.metrics['dataset']['synthetic'],
            'redis_available':app.state.cache.available,'train_metrics':scorer.metrics,**scorer.get_status()}

@app.get('/status')
def status(user=Depends(get_current_user)):
    with SessionLocal() as db: total=db.query(PredictionRecord).count()
    return {'model':'Adaptive Fraud Detection v2','fusion':app.state.scorer.get_status(),
            'total_requests':total,'train_metrics':app.state.scorer.metrics}

@app.get('/model/info')
def model_info(user=Depends(get_current_user)):
    return {**app.state.scorer.get_status(),'model_version':app.state.scorer.version,
        'persistence':{'checkpoint_count':int((config.MODEL_DIR/'online_state.pkl').exists())},
        'hyperparameters':{'alpha_stable':.8,'alpha_drift':.5,'alpha_recover':.005,'adwin_delta':.002}}

@app.post('/model/save')
def save(user=Depends(require_admin)):
    app.state.scorer.save()
    with SessionLocal() as db:log_audit(db,user.username,'model_saved')
    return app.state.scorer.get_status()

if __name__=='__main__':
    import uvicorn
    uvicorn.run('src.serving.api:app',host=config.API_HOST,port=config.API_PORT,workers=1)
