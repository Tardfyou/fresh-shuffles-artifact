"""Independently recount stored case keys, predictions, scope and live-page evidence."""
import argparse,base64,json,math
from pathlib import Path
from collections import Counter
from . import ROOT


def expected(rows,attack,smoke):
    ids=[r['sid'] for r in rows]
    if attack:return {'attack|'+s for s in (ids[:2] if smoke else ids)}
    variants=json.loads((ROOT/'protocol/PLAN.json').read_text())['tamper_types']
    return ({'functional|'+s for s in (ids[:2] if smoke else ids)}|
            {f'shared|{ids[2*i]}|{ids[2*i+1]}' for i in range(1 if smoke else len(ids)//2)}|
            {'isolation|'+s for s in (ids[:1] if smoke else ids)}|
            {f'tamper|{s}|{v}' for s in (ids[:1] if smoke else ids) for v in variants})


def compare_witness(record):
    for w in record['logit_comparisons']:
        assert math.isfinite(w['maximum_absolute_error'])
        assert abs(abs(w['worst_plain']-w['worst_observed'])-w['maximum_absolute_error'])<1e-6
        if w['exact']:assert w['maximum_absolute_error']==0 and w['plain_argmax']==w['observed_argmax']
    assert record['logits_exact']==all(w['exact'] for w in record['logit_comparisons'])


def check_consumed_inputs(run,initial):
    outputs=run['outputs'];assert set(initial)==set(outputs)
    for request,tokens in initial.items():
        assert outputs[request]['prompt_token_ids']==tokens
        observed=[sequence for step in run['bridge']['steps'] for sequence in step['sequences'] if sequence['request']==request]
        assert len(observed)==6, 'Registered greedy evaluation must execute six actual consumer calls'
        for i,sequence in enumerate(observed):assert sequence['input_ids']==tokens+outputs[request]['tokens'][:i]


def verify_job(folder,tag,block,attack=False,smoke=False):
    folder=Path(folder);inputs=[json.loads(x) for x in (ROOT/'data'/(tag+'.jsonl')).open()];by={r['sid']:r for r in inputs}
    rows=[json.loads(x) for x in (folder/'records.jsonl').open()];keys=[r['key'] for r in rows]
    assert len(keys)==len(set(keys)) and set(keys)==expected(inputs,attack,smoke),'Incomplete or duplicate case keys'
    correct=total=passes=rejections=0
    for row in rows:
        assert row['status']=='measured'
        assert row['sid']==row['key'].split('|')[1] and row['kind']==row['key'].split('|')[0]
        env=json.loads((folder/'invocations'/row['invocation']/'environment.json').read_text())
        assert env['cuda_runtime'] and env['engine']=='V0 UniProc FlashAttention eager' and env['physical_layout']['physical_block']==block
        assert env['physical_layout']['device'].startswith('cuda') and env['physical_layout']['dtype']=='torch.bfloat16'
        assert env['upstream_api']['upstream_api_pass']
        assert env['spec']['tag']==tag and env['prefix_scope_policy_installed']
        assert env['science_version']==json.loads((ROOT/'SCIENTIFIC_LOCK.json').read_text())['version']
        if row['kind']=='attack':
            truth=by[row['sid']]['attack_tokens'];truth=truth[:len(truth)//16*16];assert row['truth']==truth
            assert len(row['heads'])==2 and {h['head'] for h in row['heads']}=={0,1}
            c=n=0
            for head in row['heads']:
                assert len(head['bags'])*16==len(truth)
                counts=[sum((Counter(p)&Counter(truth[k*16:(k+1)*16])).values()) for k,p in enumerate(head['bags'])]
                assert counts==head['block_correct'] and sum(counts)==head['correct'] and head['total']==len(truth);c+=head['correct'];n+=head['total']
            assert c==row['correct'] and n==row['total'];correct+=c;total+=n
            compare_witness(row['known_key_consumer_comparison'])
            assert row['protected']['bridge']['mode']=='kvcloak' and row['protected']['bridge']['restored_objects']>0
            assert row['plain_duplicate_sanity']
            requests=list(row['protected']['outputs']);assert len(requests)==1
            check_consumed_inputs(row['protected'],{requests[0]:by[row['sid']]['attack_tokens']})
        elif row['kind']=='tamper':
            assert row['variant']==row['key'].split('|')[2] and row['input_ids']==by[row['sid']]['defense_tokens']
            expected_reject=row['rejected'] and row['model_calls_before']==row['model_calls_after']
            assert row['rejected_before_model_execution']==expected_reject
            rejections+=expected_reject
        else:
            assert row['functional_pass']==all(row['checks'].values()) and row['checks']['plain_duplicate']
            compare_witness(row['comparison']);passes+=row['functional_pass']
            protected=row['protected'];baseline=row['baseline'];requests=set(baseline['outputs']);assert requests==set(protected['outputs'])
            assert row['comparison']['tokens_equal']==all(baseline['outputs'][r]['tokens']==protected['outputs'][r]['tokens'] for r in requests)
            for r in requests:assert len(baseline['outputs'][r]['tokens'])==len(protected['outputs'][r]['tokens'])==6
            for rid,tokens in row['inputs'].items():
                sid=row['sid']
                if row['kind']=='shared' and rid.endswith('/B'):sid=row['key'].split('|')[2]
                assert tokens==by[sid]['defense_tokens']
            check_consumed_inputs(protected,row['inputs']);check_consumed_inputs(baseline,row['inputs']);check_consumed_inputs(row['duplicate_plain'],row['inputs'])
            bridge=protected['bridge'];assert bridge['mode']=='veilkv' and bridge['model_calls']>=6 and bridge['restored_objects']>0
            assert row['checks']['all_protected_roundtrips_exact']==all(s['roundtrip_bytes_exact'] for s in bridge['steps'])
            assert row['checks']['gpu_pages_zeroed_between_calls']==all(s['zeroized_at_rest'] for s in bridge['steps'])
            if row['sharing']:
                assert all(protected['outputs'][r]['cached_tokens']>=32 for r in requests)
                payloads=list(protected['payloads'].values());assert len(payloads)==2
                assert row['checks']['public_cipher_object_equal']==(base64.b64decode(payloads[0]['public'])==base64.b64decode(payloads[1]['public']))
                assert row['checks']['private_cipher_objects_differ']==(base64.b64decode(payloads[0]['private'])!=base64.b64decode(payloads[1]['private']))
    result=dict(job=folder.name,complete=True,measured=len(rows),expected=len(keys),predictions_and_scope_recounted=True,functional_passes=passes,authentication_rejections_before_model_execution=rejections,attack_correct=correct,attack_total=total,cuda_only=True)
    if smoke and not attack:
        assert passes==sum(r['kind']!='tamper' for r in rows),'Byte engine smoke had a functional negative; retain it and review before full measurement'
        assert rejections==sum(r['kind']=='tamper' for r in rows),'Authentication smoke failed; do not proceed'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();s=json.loads((ROOT/'runtime/settings.json').read_text());out=a.output or Path(s['output_dir']);out=out if out.is_absolute() else ROOT/out;reports=[]
    for folder in out.iterdir():
        if not folder.is_dir() or not (folder/'summary.json').exists():continue
        name=folder.name;tag='qwen25_05b' if 'qwen25_05b' in name else 'llama32_1b';block=int(name.rsplit('_b',1)[1]);reports.append(verify_job(folder,tag,block,'attack_' in name,name.startswith('smoke_')))
    (out/'ACCEPTANCE.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
