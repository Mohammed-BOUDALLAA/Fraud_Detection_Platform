import hashlib
import json
import redis
from redis.retry import Retry
from redis.backoff import NoBackoff
from src import config
class ShapCache:
    def __init__(self,version='unknown'):
        self.version=version
        self.client=redis.Redis(host=config.REDIS_HOST,port=config.REDIS_PORT,
            decode_responses=True,socket_connect_timeout=.1,socket_timeout=.1,retry=Retry(NoBackoff(),0))
        try: self.available=bool(self.client.ping())
        except redis.RedisError: self.available=False
    def _make_key(self,features):
        return 'shap:'+self.version+':'+hashlib.sha256(json.dumps(features,sort_keys=True).encode()).hexdigest()
    def get(self,features):
        if not self.available:return None
        try:
            cached=self.client.get(self._make_key(features))
            return json.loads(cached) if cached else None
        except (redis.RedisError,ValueError):
            self.available=False;return None
    def set(self,features,value):
        if not self.available:return
        try:self.client.set(self._make_key(features),json.dumps(value),ex=config.CACHE_TTL_SECONDS)
        except redis.RedisError:self.available=False
