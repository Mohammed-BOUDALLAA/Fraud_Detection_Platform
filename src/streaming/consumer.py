"""Forward Kafka events to the shared API; commit only successful events.
A malformed event stops the consumer without committing its offset.
Repeated delivery is safe via transaction IDs and idempotent feedback.
"""
import argparse
import json
import os
import requests
from kafka import KafkaConsumer
from src import config

def process_event(data,session,token,learn=False):
    payload={'transaction_id':data['transaction_id'],'features':data['features']}
    headers={'Authorization':f'Bearer {token}'}
    response=session.post(config.API_URL+'/predict',json=payload,headers=headers,timeout=30)
    response.raise_for_status()
    if learn and data.get('ground_truth') is not None:
        feedback=session.post(config.API_URL+'/feedback',json={'transaction_id':data['transaction_id'],
            'ground_truth':data['ground_truth']},headers=headers,timeout=30)
        feedback.raise_for_status()
    return response.json()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--learn',action='store_true');args=parser.parse_args()
    token=os.environ.get('FRAUD_API_TOKEN')
    if not token:raise SystemExit('Set FRAUD_API_TOKEN to a login token (admin for --learn).')
    consumer=KafkaConsumer(config.KAFKA_TOPIC,bootstrap_servers=config.KAFKA_BROKER,
        group_id='fraud-api-v2',enable_auto_commit=False,auto_offset_reset='earliest',
        max_poll_records=1,max_poll_interval_ms=300000,
        value_deserializer=lambda v:json.loads(v.decode()))
    try:
        with requests.Session() as session:
            for message in consumer:
                print(json.dumps(process_event(message.value,session,token,args.learn)))
                consumer.commit()
    finally:consumer.close()
if __name__=='__main__':main()
