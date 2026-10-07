"""Replay held-out processed transactions. No label flips by default."""
import argparse
import json
import time
import uuid
import joblib
import pandas as pd
from kafka import KafkaProducer
from src import config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rows',type=int,default=100)
    parser.add_argument('--delay',type=float,default=.05);args=parser.parse_args()
    df=pd.read_parquet(config.PROC_DIR/'features.parquet')
    meta=json.loads((config.PROC_DIR/'dataset_manifest.json').read_text())
    cols=joblib.load(config.MODEL_DIR/'feature_cols.pkl')
    df=df.iloc[meta['validation_end']:].head(args.rows)
    run=str(uuid.uuid4())
    producer=KafkaProducer(bootstrap_servers=config.KAFKA_BROKER,acks='all',
        value_serializer=lambda v:json.dumps(v).encode())
    try:
        for idx,row in df.iterrows():
            producer.send(config.KAFKA_TOPIC,{'transaction_id':f'{run}-{idx}',
                'features':row[cols].to_dict(),'ground_truth':int(row.isFraud)}).get(timeout=30)
            time.sleep(args.delay)
        producer.flush()
    finally:producer.close()
if __name__=='__main__':main()
