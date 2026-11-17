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
    primary={n:x for n,x in observed['main'].items()
             if x['dtype']=='torch.float32' and x['protection_dtype']=='torch.float32'
             and x['calibrator']=='corrected'}
    catalog=json.loads((root/'metadata/studies.json').read_text())
    scope=next(s for s in catalog['studies'] if s['id']=='main')['primary_reporting']
    assert set(primary)==set(scope['jobs']) and len(primary)==scope['configurations']==13
    models={json.loads((raw/n/'job.json').read_text())['model']['repo'] for n in primary}
    assert len(models)==scope['models']==11
    correct=sum(x['correct'] for x in primary.values())
    total=sum(x['total'] for x in primary.values())
    assert correct==total==scope['correct']==scope['total']==116571072
    from inspect_gpu_live602 import check as check_live
    result=check_live(root)
    result.update(primary_correct=correct,primary_total=total,primary_models=len(models),
                  primary_configurations=len(primary),main_profiles=len(primary),
                  retained_main_profiles=len(observed['main']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();print(json.dumps(check(a.root.resolve()),indent=2))
