"""Download exactly the pinned model revisions, without starting measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from . import ROOT
from .paths import model_cache


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,default=model_cache());p.add_argument('--workers',type=int,default=2);p.add_argument('--large-only',action='store_true');a=p.parse_args()
    models=json.loads((ROOT/'protocol/models.json').read_text())
    if a.large_only:models=[m for m in models if m['tag'] in ['qwen3_14b','qwen3_32b']]
    def download(m):
        path=snapshot_download(m['repo'],revision=m['revision'],cache_dir=str(a.cache_dir),max_workers=4,
            allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','merges.txt','vocab.json','LICENSE','README.md'])
        print(json.dumps(dict(model=m['repo'],revision=m['revision'],complete=True,path=path)),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(download,models))
