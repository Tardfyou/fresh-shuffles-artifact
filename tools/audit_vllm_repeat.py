"""Audit saved repeat-generation diagnostics; no engine, model or API calls."""
import argparse
import ast
import gzip
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())


def original_records(folder):
    plain=folder/'records.jsonl'
    for path in ([plain] if plain.exists() else sorted(folder.glob('records*.jsonl.gz'))):
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
            for line in f:yield json.loads(line)


def audit(diagnosis, original):
    records=diagnosis/'records';main={b:{r['sid']:r for r in original_records(original/f'vllm_baseline_b{b}')} for b in [16,32]}
    assert set(main[16])==set(main[32]) and len(main[16])==100
    for sid in main[16]:assert main[16][sid]['input_ids']==main[32][sid]['input_ids']
    expected={b:{sid for sid,r in main[b].items() if r['prediction']!=r['repeated_prediction']} for b in [16,32]}
    assert [len(expected[b]) for b in [16,32]]==[7,8]
    stage2=[];transcripts={};all_rows=0
    for b in [16,32]:
        for variant in ['A','C']:
            for run in [1,2,3]:
                d=read(records/f'stage2_{variant}_b{b}_r{run}.json');rr=d['samples'];assert len(rr)==len({r['sid'] for r in rr})==100 and {r['sid'] for r in rr}==set(main[b])
                misses=set()
                for r in rr:
                    assert r['equal']==(r['a']==r['b'])
                    assert 0<len(r['a'])<=16 and 0<len(r['b'])<=16
                    assert r['a']==main[b][r['sid']]['prediction']
                    if variant=='C':assert r['b']==main[b][r['sid']]['repeated_prediction']
                    if not r['equal']:misses.add(r['sid'])
                    else:assert r['first_div'] is None
                    if r['first_div'] is not None:
                        k=r['first_div'];assert r['a'][:k]==r['b'][:k] and r['a'][k]!=r['b'][k]
                    key=(b,variant,r['sid']);value=(r['a'],r['b']);assert key not in transcripts or transcripts[key]==value;transcripts[key]=value
                assert misses==(expected[b] if variant=='C' else set())
                assert d['equal']==100-len(misses) and d['total']==100
                stage2.append(dict(block=b,variant=variant,round=run,equal=d['equal'],total=100));all_rows+=100
    stage1=[];margins=[];swaps=0;strict_top2=0
    for f in sorted(records.glob('stage1_*.json')):
        d=read(f);rr=d['samples'];assert len(rr)==len({r['sid'] for r in rr});variant=d['variant'];b=d['block'];same=0
        for r in rr:
            assert r['sid'] in main[b] and r['in_len']==len(main[b][r['sid']]['input_ids'])
            if variant=='F':
                assert r['tokens_equal']==(r['tokens_a']==r['tokens_b'])
                assert r['prompt_lp_a']==r['prompt_lp_b'] and r['max_prompt_logprob_delta']==0.
                assert len(r['prompt_lp_a'])==r['in_len'];same+=r['tokens_equal'];continue
            a,bb=r['a']['token_ids'],r['b']['token_ids'];assert r['equal']==(a==bb);same+=r['equal']
            if r['reset'] and r['reset'].get('reset_return') is not None:assert r['reset']['reset_verified'] and r['reset']['free_blocks_after']==r['reset']['capacity']
            if variant in ['C','C0','B']:
                assert a==main[b][r['sid']]['prediction'] and bb==main[b][r['sid']]['repeated_prediction']
            if variant in ['A','D','E']:assert a==bb
            if variant=='C' and a!=bb:
                k=r['first_div'];assert a[:k]==bb[:k] and a[k]!=bb[k]
                first=r['a']['logprobs'][k];second=r['b']['logprobs'][k]
                top_groups=[{x[0] for x in values if x[1]>=values[1][1]-1e-6} for values in [first,second]]
                assert {a[k],bb[k]}<=top_groups[0] and {a[k],bb[k]}<=top_groups[1]
                strict_top2+=int({a[k],bb[k]}<={x[0] for x in first[:2]} and {a[k],bb[k]}<={x[0] for x in second[:2]})
                margins.extend([first[0][1]-first[1][1],second[0][1]-second[1][1]]);swaps+=1
        stage1.append(dict(file=f.name,variant_label=variant,executed_condition='C: reset once before pair' if variant=='B' else variant,exact_outputs=same,total=len(rr)));all_rows+=len(rr)
    # Inspect the submitted source, instead of trusting the report's B label.
    source=diagnosis/'scripts/diag_stage1.py'
    if not source.exists():
        source=next(parent/'sources/vllm-repeat-diagnosis/scripts/diag_stage1.py' for parent in diagnosis.parents if (parent/'sources/vllm-repeat-diagnosis/scripts/diag_stage1.py').exists())
    tree=ast.parse(source.read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='VARIANT' and isinstance(n.test.comparators[0],ast.Tuple) and any(isinstance(x,ast.Constant) and x.value=='B' for x in n.test.comparators[0].elts))
    resets=[n for line in branch.body for n in ast.walk(line) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='reset_cache'];assert len(resets)==1
    assert swaps==15 and {round(x,6) for x in margins}=={0.,.0625,.125} and strict_top2==14
    return dict(record_integrity_pass=True,stage2_pairs=1200,all_recorded_pairs=all_rows,stage2=stage2,stage1=stage1,near_tie_first_divergences=swaps,strict_first_two_list_entries_contain_both_winners=strict_top2,remaining_event_has_tied_second_score_group=True,observed_top2_logprob_gaps=sorted({round(x,6) for x in margins}),stage2_full_transcripts_exact_to_original=True,cache_disabled_all_full_cohort_pairs_equal=True,stage1_B_label_valid=False,stage1_B_actual_condition='Reset once before pair; same branch as C, not a cold-cold intervention',cold_cold_effect_not_tested=True,original_baseline_requires_replacement=False,reason_scope='Repeatable prefix-cache-associated BF16 greedy near-tie sensitivity on this cohort; exact internal computation not isolated',no_relaxation_of_live_defense_correctness=True,concurrency_increases_pinned_failure_probability='not established by these records')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=audit(a.diagnosis,a.original);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
