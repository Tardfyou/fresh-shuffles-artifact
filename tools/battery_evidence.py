"""Read retained defense-battery results without manuscript dependencies."""
import json
import math

def summarize(results, pending, require):
    stages={'fixation','functional','tee_battery','paged_harness','attacks','perf','adjudicate'}
    codes=results/'battery_exitcodes.json'
    if not codes.exists():raise pending('battery stage completion file missing')
    codes=json.loads(codes.read_text())
    if set(codes)!=stages:raise pending(f'{len(codes)}/7 battery stages recorded')
    require(all(v==0 for v in codes.values()),'battery stage failed')
    directory=results/'battery'
    if (results/'battery_perf512/INPUT_SCOPE.json').exists():
        directory=results/'battery_perf512'
        supplement=directory/'exitcodes.json'
        if not supplement.exists():raise pending('512-token timing rerun is active')
        supplement=json.loads(supplement.read_text())
        if set(supplement)!={'perf','adjudicate'}:raise pending('512-token timing rerun is active')
        require(all(v==0 for v in supplement.values()),'512-token timing rerun failed')
    data=json.loads((directory/'results/veilkv_plus_v2.json').read_text())
    if directory.name=='battery_perf512':
        original=json.loads((results/'battery/results/veilkv_plus_v2.json').read_text())
        require(all(data[name]==original[name] for name in ['fixation','functional','tee_battery','paged_harness','attacks']),
                'reused successful stage records differ from their original source')
    require(set(data['stages_done'])==stages,'battery stage inventory differs')
    entries=[]
    for model in ['Qwen2.5-0.5B','Llama-3.2-1B']:
        cells=data['functional'][model]
        for name in ['sys+secret_tee_float32','sys+secret_tee_bfloat16','unaligned_tee_float32','unaligned_tee_bfloat16']:
            cell=cells[name]
            require(all(math.isfinite(cell[k]) for k in ['roundtrip_K','roundtrip_V']),'nonfinite functional error')
            require(isinstance(cell['greedy20_equal'],bool),'continuation outcome missing')
            entries.append(dict(model=model,name=name,**cell))
    require(isinstance(data['attacks'],list),'legacy inherited battery output is not a fresh result')
    dictionary=[r for r in data['attacks'] if r['attack']=='private_dictionary_100']
    require(len(dictionary)==2 and {r['model'] for r in dictionary}=={'Qwen2.5-0.5B','Llama-3.2-1B'},'two model measurements required')
    require(all(r['denominator']==100 and 0<=r['numerator']<=100 for r in dictionary),'invalid candidate counters')
    perf=data['perf']
    for key in ['protect_tee','recover_tee','protect_kvcloak','recover_kvcloak','protect_nosharing_aead']:
        require(all(math.isfinite(perf[key][q]) and perf[key][q]>=0 for q in ['p50','p95','p99','mean','std']),'invalid timing distribution')
    measurement=perf['measurement']
    require(measurement['actual_tokens']>0 and measurement['actual_tokens']%16==0,'actual benchmark length missing')
    require(measurement['input_clone_outside_timing'] and measurement['kvcloak_fused'] is False,'timing input protocol differs')
    if directory.name=='battery_perf512':
        require(measurement['actual_tokens']==512 and measurement['public_prefix_tokens']==32 and measurement['private_tail_tokens']==480,'wrong measured cache layout')
        import numpy as np
        for key in ['protect_tee','recover_tee','protect_kvcloak','recover_kvcloak','protect_nosharing_aead']:
            values=perf[key]['samples_seconds_per_1000_tokens']
            require(len(values)==100,'raw timing repetitions are incomplete')
            expected=dict(p50=np.quantile(values,.5),p95=np.quantile(values,.95),p99=np.quantile(values,.99),mean=np.mean(values),std=np.std(values))
            require(all(abs(perf[key][q]-value)<=.000051 for q,value in expected.items()),'timing summary differs from raw observations')
    context=json.loads((directory/'performance_context.json').read_text())
    require(context['own_experiment_workers']==0,'project timing measurements overlapped experiment workers')
    return dict(stages=7,functional=entries,tee=data['tee_battery'],paged=data['paged_harness'],
                arithmetic_dictionary=[dict(model=r['model'],correct=r['numerator'],total=r['denominator']) for r in dictionary],
                perf=perf,performance_context=context,adjudication=data['adjudication'],timing_source=str(directory.relative_to(results)))
