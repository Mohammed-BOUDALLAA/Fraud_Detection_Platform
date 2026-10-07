import os
import secrets
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')
DATA_DIR = Path(os.getenv('FRAUD_DATA_DIR', str(BASE_DIR / 'data')))
RAW_DIR, PROC_DIR, MODEL_DIR = [DATA_DIR / x for x in ('raw','processed','models')]
LOG_DIR = BASE_DIR / 'logs'
for folder in (RAW_DIR, PROC_DIR, MODEL_DIR, LOG_DIR): folder.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///' + str(DATA_DIR / 'app.db'))
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY') or secrets.token_urlsafe(48)
JWT_ALGORITHM = 'HS256'
JWT_EXPIRE_MINUTES = 60
JWT_REMEMBER_ME_MINUTES = 1440
KAFKA_BROKER = os.getenv('KAFKA_BROKER','localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC','transactions')
REDIS_HOST = os.getenv('REDIS_HOST','localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT','6379'))
CACHE_TTL_SECONDS = 3600
API_HOST = os.getenv('API_HOST','127.0.0.1')
API_PORT = int(os.getenv('API_PORT','8000'))
API_URL = os.getenv('API_URL',f'http://127.0.0.1:{API_PORT}')
RIVER_SAVE_EVERY_N_UPDATES = 100
RIVER_AUTOSAVE_INTERVAL_SECONDS = 300
RIVER_MAX_CHECKPOINTS = 5
LOG_LEVEL = os.getenv('LOG_LEVEL','INFO')
