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
    live=root/'results/gpu/accepted/vllm-live-a100-20261110-01'
    sys.path.insert(0,str(root/'sources/vllm-live'))
    from integration.verify import verify_job
    reports=[];max_defense_error=0.;attack_output_equal=0
    # The unchanged issued verifier reads uncompressed records; materialize one
    # job at a time, leaving the compressed publication and private data intact.
    with tempfile.TemporaryDirectory(prefix='veilkv-offline-live-') as temporary:
        for job in sorted(live.iterdir()):
            if not job.is_dir():continue
            folder=Path(temporary)/job.name;folder.mkdir()
            shutil.copytree(job/'invocations',folder/'invocations')
            with (folder/'records.jsonl').open('w') as out:
                for r in records(job):
                    out.write(json.dumps(r)+'\n')
                    if job.name.startswith('smoke_'):continue
                    if r['kind']=='attack':attack_output_equal+=r['known_key_consumer_comparison']['tokens_equal']
                    elif 'comparison' in r:
                        max_defense_error=max(max_defense_error,max(w['maximum_absolute_error'] for w in r['comparison']['logit_comparisons']))
            tag='qwen25_05b' if 'qwen25_05b' in job.name else 'llama32_1b'
            reports.append(verify_job(folder,tag,int(job.name.rsplit('_b',1)[1]),'attack_' in job.name,job.name.startswith('smoke_')))
            shutil.rmtree(folder)
    formal=[r for r in reports if not r['job'].startswith('smoke_')]
    assert len(reports)==10 and sum(r['measured'] for r in reports if r['job'].startswith('smoke_'))==50
    assert sum(r['measured'] for r in formal)==780
    assert sum(r['functional_passes'] for r in formal)==520
    assert sum(r['authentication_rejections_before_model_execution'] for r in formal)==160
    assert sum(r['attack_correct'] for r in formal)==sum(r['attack_total'] for r in formal)==41344
    assert max_defense_error==0. and attack_output_equal==90
    return dict(pass_saved_records=True,primary_correct=91611072,primary_total=91611072,main_profiles=len(observed['main']),live_formal_cases=780,live_smoke_cases=50,live_defense_functional=520,live_rejections=160,live_attack_correct=41344,live_attack_total=41344,live_attack_adapter_output_equal=90,model_or_api_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();print(json.dumps(check(a.root.resolve()),indent=2))
