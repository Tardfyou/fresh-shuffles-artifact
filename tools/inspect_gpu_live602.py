"""Recount accepted live602 case keys, predictions and consumer witnesses offline."""
import argparse,collections,json,shutil,sys,tempfile
from pathlib import Path
from summarize_gpu_results import records

def check(root):
    index=json.loads((root/'results/gpu/index.json').read_text());run=index['primary_live_run']
    entry=next(x for x in index['runs'] if x['run_id']==run);live=root/entry['raw_path'];accepted=json.loads((root/entry['validation_path']).read_text())
    sys.path.insert(0,str(root/'sources/vllm-live'))
    from integration.verify import verify_job
    targets=set(json.loads((root/'metadata/gpu602/target602_ids.json').read_text()));assert len(targets)==602
    for tag in ['qwen25_05b','llama32_1b']:
        inputs=list(map(json.loads,(root/'sources/vllm-live/data'/(tag+'.jsonl')).open()));assert {r['sid'] for r in inputs}==targets
    reports=[];kinds=collections.Counter();adapter_equal=0;defense_error=0.;adapter_error=0.;epochs=0
    with tempfile.TemporaryDirectory(prefix='live602-record-check-') as temporary:
        for job in sorted(live.iterdir()):
            if not job.is_dir() or not list(job.glob('records*.jsonl.gz')):continue
            folder=Path(temporary)/job.name;folder.mkdir();shutil.copytree(job/'invocations',folder/'invocations')
            with (folder/'records.jsonl').open('w') as stream:
                for row in records(job):
                    stream.write(json.dumps(row)+'\n')
                    if job.name.startswith('smoke_'):continue
                    kinds[row['kind']]+=1
                    if row['kind']=='attack':
                        adapter_equal+=row['known_key_consumer_comparison']['tokens_equal']
                        adapter_error=max(adapter_error,max(w['maximum_absolute_error'] for w in row['known_key_consumer_comparison']['logit_comparisons']))
                    elif row['kind']!='tamper':
                        defense_error=max(defense_error,max(w['maximum_absolute_error'] for w in row['comparison']['logit_comparisons']))
                        epochs+=bool(row['checks'].get('real_epoch_transition',False))
                        if row['sharing']:
                            protected=row['protected'];ids=list(protected['outputs']);assert len(ids)==2
                            steps=[s for s in protected['bridge']['steps'] if set(ids)<={q['request'] for q in s['sequences']}];assert steps
                            n=32//int(job.name.rsplit('_b',1)[1])
                            for step in steps:
                                by={q['request']:q for q in step['sequences']};a,b=by[ids[0]],by[ids[1]]
                                assert a['physical_blocks'][:n]==b['physical_blocks'][:n]
                                assert not(set(a['physical_blocks'][n:])&set(b['physical_blocks'][n:]))
            tag='qwen25_05b' if 'qwen25' in job.name else 'llama32_1b';block=int(job.name.rsplit('_b',1)[1])
            reports.append(verify_job(folder,tag,block,'attack_' in job.name,job.name.startswith('smoke_')));shutil.rmtree(folder)
    formal=[r for r in reports if not r['job'].startswith('smoke_')];smoke=[r for r in reports if r['job'].startswith('smoke_')]
    assert len(formal)==len(smoke)==5 and sum(r['measured'] for r in smoke)==50
    assert sum(r['measured'] for r in formal)==accepted['formal_cases']==25886
    assert sum(r['functional_passes'] for r in formal)==accepted['functional_passes']==6020
    assert sum(r['authentication_rejections_before_model_execution'] for r in formal)==accepted['rejections']==19264
    assert sum(r['attack_correct'] for r in formal)==sum(r['attack_total'] for r in formal)==accepted['attack_correct']==accepted['attack_total']==359936
    assert kinds==collections.Counter(accepted['formal_kinds'])
    assert adapter_equal==accepted['attack_adapter_output_equal']==562 and epochs==40
    assert defense_error==accepted['worst_defense_logit_error']==0.
    assert adapter_error==accepted['worst_attack_adapter_logit_error']
    return dict(pass_saved_records=True,live_run=run,live_formal_cases=25886,live_smoke_cases=50,live_defense_functional=6020,live_rejections=19264,live_attack_correct=359936,live_attack_total=359936,live_attack_adapter_output_equal=562,live_attack_adapter_trials=602,epoch_transition_cases=40,actual_physical_sharing_and_isolation_pass=True,model_or_api_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--report',type=Path);a=p.parse_args();result=check(a.root.resolve())
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
