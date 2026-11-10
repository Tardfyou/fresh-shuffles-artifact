"""Recount the fixed-cohort CPU large-model extension without inference or APIs."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np


def inspect(root):
    source=root/'results/cpu_large_models_20261110'
    manifest=[json.loads(x) for x in (root/'results/main_public/samples_main_public.jsonl').open()]
    expected=[r['sid'] for r in manifest];strata=defaultdict(list)
    for i,row in enumerate(manifest):strata[row['len_bucket']].append(i)
    models={}
    for tag in ['qwen3_14b','qwen3_32b']:
        rows=[json.loads(x) for x in (source/tag/'per_sample.jsonl').open()]
        summary=json.loads((source/tag/'summary.json').read_text())
        assert [r['sid'] for r in rows]==expected and len(set(expected))==3000
        cs=[];ns=[]
        for row in rows:
            assert row['status']=='measured' and set(row['seeds'])=={'5001','5002'}
            c=n=0;truth=row['truth']
            for scores in row['seeds'].values():
                assert len(scores['heads'])==8 and {h['head'] for h in scores['heads']}==set(range(8))
                hc=hn=0
                for head in scores['heads']:
                    assert head['total']==len(truth) and len(head['bags'])*16==len(truth)
                    counts=[sum((Counter(b)&Counter(truth[16*k:16*k+16])).values()) for k,b in enumerate(head['bags'])]
                    assert counts==head['block_correct'] and sum(counts)==head['correct']
                    hc+=head['correct'];hn+=head['total']
                assert (hc,hn)==(scores['correct'],scores['total']);c+=hc;n+=hn
            cs.append(c);ns.append(n)
        c=np.asarray(cs,dtype=np.int64);n=np.asarray(ns,dtype=np.int64)
        assert (int(c.sum()),int(n.sum()))==(summary['correct'],summary['total'])
        rng=np.random.default_rng(20261029);rates=[]
        for _ in range(2000):
            index=np.concatenate([rng.choice(group,len(group),replace=True) for group in strata.values()]);rates.append(c[index].sum()/n[index].sum())
        ci=(100*np.quantile(rates,[.025,.975])).tolist();assert ci==summary['bootstrap95_percent']
        models[tag]=dict(prompts=3000,correct=int(c.sum()),total=int(n.sum()),recovery_percent=100*c.sum()/n.sum(),bootstrap95_percent=ci,normal_wrong_outputs_retained=True)
    return dict(large_model_cpu_checks_pass=True,models=models,model_or_api_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();print(json.dumps(inspect(a.root),indent=2))
