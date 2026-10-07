import json
import shutil
from pathlib import Path
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.serving.api import app
from src.auth.database import Base, engine, SessionLocal
from src.auth.models import Role
from src.users.crud import create_user
from src.models.feature_engineering import FeatureBuilder
from src.models.fusion import AdaptiveFusionScorer
from src import config

@pytest.fixture(scope='module')
def client():
    # Test database is selected by FRAUD_DATA_DIR in the test runner.
    with TestClient(app) as client:
        with SessionLocal() as db:
            for name,role in [('testadmin',Role.ADMIN),('testanalyst',Role.ANALYST)]:
                if not db.query(__import__('src.auth.models',fromlist=['User']).User).filter_by(username=name).first():
                    create_user(db,name,'Test-password-42',role)
        yield client

@pytest.fixture
def payload():
    cols=joblib.load(config.MODEL_DIR/'feature_cols.pkl')
    row=pd.read_parquet(config.PROC_DIR/'features.parquet').iloc[-1]
    return {'features':row[cols].to_dict()}

def auth(client,name='testadmin'):
    r=client.post('/auth/login',json={'username':name,'password':'Test-password-42'})
    assert r.status_code==200
    return {'Authorization':'Bearer '+r.json()['access_token']}

def test_auth_and_health(client,payload):
    assert client.get('/health').status_code==200
    assert client.post('/predict',json=payload).status_code==401
    assert client.post('/auth/login',json={'username':'testadmin','password':'wrong'}).status_code==401
    assert client.get('/auth/me',headers=auth(client)).status_code==200

def test_prediction_explanation_and_no_learning(client,payload):
    h=auth(client)
    before=client.get('/model/info',headers=h).json()['n_updates']
    r=client.post('/predict',json=payload,headers=h)
    assert r.status_code==200,r.text
    result=r.json()
    assert 0<=result['fused_score']<=1
    assert len(result['shap_top3'])==3
    assert result['fused_score']==pytest.approx(result['alpha']*result['xgb_score']+(1-result['alpha'])*result['river_score'])
    assert client.get('/model/info',headers=h).json()['n_updates']==before

def test_invalid_inputs(client,payload):
    h=auth(client)
    assert client.post('/predict',headers=h,json={'features':{}}).status_code==422
    assert client.post('/predict',headers=h,json={**payload,'ground_truth':1}).status_code==422
    assert client.post('/predict',headers=h,json={'features':{**payload['features'],'isFraud':1}}).status_code==422
    assert client.post('/feedback',headers=h,json={'transaction_id':'none','ground_truth':2}).status_code==422
    assert client.post('/predict/batch',headers=h,json=[]).status_code==422

def test_feedback_once_and_persistence(client,payload):
    h=auth(client)
    result=client.post('/predict',headers=h,json=payload).json()
    body={'transaction_id':result['transaction_id'],'ground_truth':1}
    assert client.post('/feedback',headers=auth(client,'testanalyst'),json=body).status_code==403
    first=client.post('/feedback',headers=h,json=body)
    assert first.status_code==200,first.text
    assert first.json()['updated'] is True
    assert client.post('/feedback',headers=h,json=body).json()['updated'] is False
    restored=AdaptiveFusionScorer()
    assert result['transaction_id'] in restored.feedback_ids
    assert restored.n_updates==first.json()['n_updates']

def test_idempotency_and_batch(client,payload):
    h=auth(client)
    body={**payload,'transaction_id':'idempotency-test'}
    first=client.post('/predict',headers=h,json=body)
    assert client.post('/predict',headers=h,json=body).json()==first.json()
    altered={**body,'features':{**body['features'],'TransactionAmt':999.}}
    assert client.post('/predict',headers=h,json=altered).status_code==409
    assert len(client.post('/predict/batch',headers=h,json=[payload,payload]).json())==2

def test_preprocessing_fit_only_train():
    train=pd.DataFrame({'TransactionID':[1,2],'TransactionDT':[1,2],'TransactionAmt':[10.,20.],
        'card1':[1,1],'ProductCD':['W','C'],'isFraud':[0,1]})
    builder=FeatureBuilder().fit(train)
    test=train.iloc[:1].copy();test['TransactionAmt']=1000.;test['ProductCD']='unseen'
    out=builder.transform(test)
    assert out.amt_vs_card_mean.iloc[0]==pytest.approx(1000/16)
    assert out.ProductCD.iloc[0]==-1
    assert 'isFraud' not in out and 'TransactionID' not in out

def test_raw_processed_parity(client):
    h=auth(client)
    raw=pd.read_csv(config.DATA_DIR/'demo_raw/train_transaction.csv').iloc[-1].drop('isFraud').to_dict()
    processed=app.state.scorer.preprocessor.transform(pd.DataFrame([raw])).iloc[0].to_dict()
    a=client.post('/predict',headers=h,json={'features':raw,'input_type':'raw'})
    b=client.post('/predict',headers=h,json={'features':processed})
    assert a.status_code==b.status_code==200
    assert a.json()['fused_score']==pytest.approx(b.json()['fused_score'])

def test_redis_failure_fallback():
    from src.serving.cache import ShapCache
    import redis
    from unittest.mock import Mock
    cache=ShapCache.__new__(ShapCache);cache.available=True;cache.version='test';cache.client=Mock()
    cache.client.get.side_effect=redis.ConnectionError('offline')
    assert cache.get({'a':1.}) is None
    assert not cache.available

def test_streaming_forwarding_and_http_failure():
    from src.streaming.consumer import process_event
    from unittest.mock import Mock
    import requests
    session=Mock();session.post.return_value.json.return_value={'label':0}
    event={'transaction_id':'stream-one','features':{'a':1},'ground_truth':0}
    assert process_event(event,session,'token',learn=True)=={'label':0}
    assert session.post.call_count==2
    session.post.return_value.raise_for_status.side_effect=requests.HTTPError('failure')
    with pytest.raises(requests.HTTPError):process_event(event,session,'token',learn=True)

def test_drift_response_and_recovery():
    from unittest.mock import Mock
    scorer=AdaptiveFusionScorer()
    scorer.adwin=Mock();scorer.adwin.drift_detected=True
    scorer.save=Mock()
    row=pd.read_parquet(config.PROC_DIR/'features.parquet').iloc[0][scorer.feature_cols].to_dict()
    scorer.learn_feedback('drift-test',row,1,0)
    assert scorer.alpha==.5 and scorer.in_drift
    scorer.adwin.drift_detected=False
    scorer.learn_feedback('recovery-test',row,0,0)
    assert scorer.alpha==pytest.approx(.505)

def test_dashboard_login_and_pages(client,monkeypatch):
    import requests
    from streamlit.testing.v1 import AppTest
    def bridge(method,url,**kwargs):
        kwargs.pop('timeout',None)
        return client.request(method,url,**kwargs)
    monkeypatch.setattr(requests,'request',bridge)
    dashboard=AppTest.from_file(str(config.BASE_DIR/'src/serving/demo.py')).run(timeout=20)
    assert len(dashboard.exception)==0
    dashboard.session_state['token']=auth(client)['Authorization'].split(' ',1)[1]
    dashboard.session_state['role']='admin'
    dashboard.run(timeout=20)
    assert len(dashboard.exception)==0
    assert len(dashboard.tabs)==4

def test_cli_preprocessing_artifact_portability(tmp_path):
    import os
    import subprocess
    import sys
    raw=tmp_path/'raw';raw.mkdir()
    source=config.DATA_DIR/'demo_raw'
    for name in ('train_transaction.csv','train_identity.csv'):shutil.copy(source/name,raw/name)
    env={**os.environ,'FRAUD_DATA_DIR':str(tmp_path)}
    run=subprocess.run([sys.executable,'-m','src.models.feature_engineering'],env=env,cwd=config.BASE_DIR,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    builder=joblib.load(tmp_path/'models/preprocessor.pkl')
    assert type(builder).__module__=='src.models.feature_engineering'
