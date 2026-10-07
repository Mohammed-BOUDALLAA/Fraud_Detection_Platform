"""Use an isolated copy of demo data: never mutate the user's online state or DB."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temp:
    data=Path(temp)/'data'
    shutil.copytree(root/'data',data,ignore=shutil.ignore_patterns('app.db*','online_state.pkl'))
    env={**os.environ,'FRAUD_DATA_DIR':str(data),'DATABASE_URL':'sqlite:///'+str(data/'test.db')}
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q'],cwd=root,env=env))
