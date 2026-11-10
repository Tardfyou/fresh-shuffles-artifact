"""Recount accepted GPU observations without model or API calls."""
import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path


def records(folder):
    files = [folder / 'records.jsonl'] if (folder / 'records.jsonl').exists() else sorted(folder.glob('records*.jsonl.gz'))
    for file in files:
        opener = gzip.open if file.suffix == '.gz' else open
        with opener(file, 'rt') as stream:
            for line in stream:
                yield json.loads(line)


def score(items):
    items = list(items)
    c, n = sum(x['correct'] for x in items), sum(x['total'] for x in items)
    return dict(correct=c, total=n, percent=100*c/n if n else None)


def summarize(root):
    out = dict(main={}, calibration={}, epochs={}, ratios=[], ordering={}, matched={}, spectrum={}, latency={}, sa={})
    for folder in sorted(root.iterdir()):
        name = folder.name
        if not folder.is_dir() or name.startswith('smoke_'): continue
        if name.startswith('gpu_main_'):
            c = n = 0
            categories = collections.defaultdict(lambda: [0, 0])
            for r in records(folder):
                assert r['status'] == 'measured'
                for s in r['seeds'].values(): c += s['correct']; n += s['total']
            env=json.loads((folder/'environment.json').read_text())
            cal=json.loads((folder/'calibration.json').read_text())
            out['main'][name]=dict(correct=c,total=n,percent=100*c/n,probe=cal['actual_probe_tokens'],heads=env['heads'],d=env['head_dimension'],block=env['block_size'],dtype=env['model_dtype'],protection_dtype=env['cloak_dtype'],calibrator=env['calibrator'],scope=env['strategy'])
            out['calibration'][name]=dict(heads=cal['heads'])
        elif name == 'gpu_probe':
            rr=list(records(folder)); out['probe']=[dict(budget=b,coverage=sum(r['coverage'] for r in rr if r['budget']==b),calibration=sum(r['calibration'] for r in rr if r['budget']==b),n=sum(r['budget']==b for r in rr)) for b in sorted({r['budget'] for r in rr})]
        elif name == 'gpu_fused': out['fused']=score(r['scores'] for r in records(folder))
        elif name in ['gpu_lifecycle','gpu_sa_rotation']:
            rr=list(records(folder));out[name.removeprefix('gpu_')]={k:score(r['conditions'][k] for r in rr) for k in rr[0]['conditions']}
        elif name.startswith('gpu_cross_key_'):
            rr=list(records(folder));tag=name.removeprefix('gpu_cross_key_');out['epochs'][tag]={'by_key':{}}
            for key in sorted({r['key_seed'] for r in rr}):
                group=[r for r in rr if r['key_seed']==key];v=score(r['scores'] for r in group);v['minimum_prompt_percent']=min(100*r['scores']['correct']/r['scores']['total'] for r in group);out['epochs'][tag]['by_key'][str(key)]=v
        elif name.startswith('gpu_ratio_'):
            rr=list(records(folder));v={k:score(r['scores'][k] for r in rr) for k in ['same_epoch','adapted']}
            deltas=[r['scores']['adapted']['correct']-r['scores']['same_epoch']['correct'] for r in rr]
            v.update(model='Qwen2.5-0.5B' if 'qwen' in name else 'Llama-3.2-1B',ratio=rr[0]['ratio'],net_head_token_gain=sum(deltas),gain_percentage_points=v['adapted']['percent']-v['same_epoch']['percent'],sample_ids_improved=sum(d>0 for d in deltas),sample_ids_unchanged=sum(d==0 for d in deltas),sample_ids_degraded=sum(d<0 for d in deltas));out['ratios'].append(v)
        elif name == 'gpu_ordering':
            rr=list(records(folder));out['ordering']['n']=len(rr)
            for k in rr[0]['conditions']:
                values=[r['conditions'][k] for r in rr];out['ordering'][k]=dict(macro_position_percent=statistics.mean(100*x['position_correct']/x['tokens'] for x in values),exact_spans=sum(x['exact_span'] for x in values))
            out['ordering']['exact_bags']=sum(r['bag_source']['correct']==r['bag_source']['total'] for r in rr)
        elif name.startswith(('gpu_matched_qwen','gpu_matched_llama','gpu_spectrum_')):
            rr=list(records(folder));section='spectrum' if 'spectrum_' in name else 'matched';tag='qwen25_05b' if 'qwen' in name else 'llama32_1b'
            out[section][tag]={rule:{seed:sum(r['seeds'][seed][rule]['hit'] for r in rr) for seed in rr[0]['seeds']} for rule in rr[0]['seeds']['5001']}
        elif name == 'gpu_privacy':
            rr=list(records(folder));out['privacy']={label:{k:100*statistics.mean(r['identification'][label]['topk'][k] for r in rr) for k in ['1','5','10']} for label in ['same_epoch','wrong_secret']}
        elif name.startswith('gpu_matched_latency_'):
            groups=collections.defaultdict(list)
            for r in records(folder):groups[r['method']+'|'+r['operation']].append(r)
            out['latency'][name]={k:dict(median_seconds=statistics.median(r['seconds'] for r in rr),median_seconds_per1000=statistics.median(r.get('seconds_per_1000_tokens',r['seconds']*1000/512) for r in rr),n=len(rr),samples_seconds_per_1000_tokens=[r.get('seconds_per_1000_tokens',r['seconds']*1000/512) for r in rr]) for k,rr in groups.items()}
        elif name.startswith('gpu_sa_functional_'):
            dtype='bf16' if '_bfloat16_' in name else 'FP32'
            for r in records(folder):
                key=dtype+'|'+r['condition'];v=out['sa'].setdefault(key,dict(passes=0,total=0));v['passes']+=sum(s['functional_pass'] for s in r['segments']);v['total']+=len(r['segments'])
        elif name.startswith('gpu_calibration_cost_'):
            out.setdefault('session_cost',{})[name.removeprefix('gpu_calibration_cost_')]=statistics.median(r['seconds'] for r in records(folder))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();value=summarize(a.results);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n');print(a.output)
