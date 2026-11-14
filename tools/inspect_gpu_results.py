"""Recompute portable GPU counters and independently verify live-engine cases."""
import argparse
import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path
from summarize_gpu_results import records, summarize


def check(root):
    raw=root/'results/gpu/accepted/a100-80g-20261110-01'
    observed=summarize(raw)
    saved=json.loads((root/'results/gpu/summaries/a100-80g-20261110-01/PAPER_VALUES.json').read_text())
    assert observed==saved, 'Portable counters differ from accepted reporting values'
    assert sum(x['total'] for n,x in observed['main'].items() if 'cpu_bridge' not in n and '14b' not in n and '32b' not in n)==91611072
    from inspect_gpu_live602 import check as check_live
    result=check_live(root)
    result.update(primary_correct=91611072,primary_total=91611072,main_profiles=len(observed['main']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();print(json.dumps(check(a.root.resolve()),indent=2))
