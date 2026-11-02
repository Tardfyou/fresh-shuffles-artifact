"""Independent frozen-key/field/fixture audit for known-key defense evidence."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path


def read(path):return json.loads(path.read_text())


def expected_checks(condition):
    common={'independent_plain_sanity','plain_source_unmodified'}
    negatives={'principal_only','nonce_only','epoch_only','dtype_only','profile_only','length_only','prefix_token_only','public_policy_only','ciphertext_byte','truncation'}
    if condition=='chained_reencode':
        result=common|negatives|{'duplicate_nonce'}
        for i in range(3):result|={f'hop{i}_input_unmodified',f'hop{i}_roundtrip',f'hop{i}_plain_cache',f'hop{i}_epoch_only',f'seg{i+1}_logits',f'seg{i+1}_cache'}
        return result
    if condition=='identity_epoch':return common|negatives|{'A_input_unmodified','A_roundtrip','B_input_unmodified','B_roundtrip','B_equals_plain_boundary','A_principal_only','B_principal_only','seg1_logits','seg1_cache','seg2_logits','seg2_cache'}
    if condition=='mid_session':return common|negatives|{'mid_input_unmodified','mid_roundtrip','seg2_logits','seg2_cache','seg3_logits','seg3_cache'}
    return common|{'prefixA_input_unmodified','prefixA_roundtrip','prefixB_input_unmodified','prefixB_roundtrip','public_identity_all_layers','public_payload_all_layers','stored_prefix_recovery','shared_logits','shared_cache','shared_principal_only','shared_prefix_only'}


def audit(root,require_complete=False):
    rows=[json.loads(l) for l in (root/'inputs.jsonl').open()]
    ids=[r['sid'] for r in rows];assert len(ids)==len(set(ids))==602
    manifest={r['sid']:r for r in map(json.loads,(root.parents[2]/'main_public/samples_main_public.jsonl').open())}
    assert set(ids)=={sid for sid,r in manifest.items() if r['or_subset']}
    assert all(r==manifest[r['sid']] for r in rows)
    models=['Qwen2.5-0.5B','Llama-3.2-1B'];dtypes=['float32','bfloat16'];shapes=['no_pad','left8','right8','left16','right16'];conditions=['chained_reencode','identity_epoch','mid_session','shared_prefix']
    expected={f'{sid}|{model}|{dtype}|{shape}|{condition}' for sid in ids for model in models for dtype in dtypes for shape in shapes for condition in conditions}
    expected_segments={'chained_reencode':[1,2,3],'identity_epoch':[1,2],'mid_session':[2,3],'shared_prefix':[1]}
    seen=set();counts=Counter();cells=defaultdict(Counter);issues=[];functional_failures=[];full_errors=defaultdict(float);source_checks={}
    fields={'principal_only':'principal','A_principal_only':'principal','B_principal_only':'principal','shared_principal_only':'principal','nonce_only':'nonce','epoch_only':'epoch_id','dtype_only':'storage_dtype','profile_only':'profile','length_only':'true_len','prefix_token_only':'tokens','shared_prefix_only':'tokens','public_policy_only':'n_prefix_blocks'}
    for file in sorted((root/'full').glob('*/checks.jsonl')):
        start_size=file.stat().st_size
        for lineno,line in enumerate(file.open(),1):
            if not line.endswith('\n') and not require_complete:break
            try:row=json.loads(line)
            except json.JSONDecodeError:issues.append([str(file.relative_to(root)),lineno,'invalid_json']);continue
            key=row['key'];counts['records']+=1
            if key in seen:issues.append([key,'duplicate'])
            seen.add(key)
            if key not in expected:issues.append([key,'unexpected_key']);continue
            sid,model,dtype,shape,condition=key.split('|')
            for name,value in [('sid',sid),('model',model),('dtype',dtype),('shape',shape),('condition',condition)]:
                if row.get(name)!=value:issues.append([key,'metadata_'+name])
            cell=cells[f'{model}|{dtype}|{shape}|{condition}'];cell['records']+=1
            if row['status']!='measured':issues.append([key,row.get('error_type'),row.get('error')]);counts['errors']+=1;continue
            check=row['checks'];wanted=expected_checks(condition)
            if set(check)!=wanted:issues.append([key,'check_set',sorted(wanted-set(check)),sorted(set(check)-wanted)])
            passing=bool(check) and all(v is True for v in check.values())
            if passing!=row['functional_pass']:issues.append([key,'pass_flag_disagrees'])
            if not passing:functional_failures.append([key,'functional_failure',[k for k,v in check.items() if v is not True]])
            counts['functional_pass']+=passing;cell['functional_pass']+=passing
            segments=row['segments']
            if [s['segment'] for s in segments]!=expected_segments[condition]:issues.append([key,'segment_set'])
            for seg in segments:
                lg=seg['logits'];good=lg.get('exact') is True and lg.get('shape_equal') is True and lg.get('finite') is True and lg.get('max')==0 and lg.get('argmax_equal') is True and seg.get('cache_mismatch')==[]
                counts['consumer_comparisons']+=1;counts['consumer_comparisons_passed']+=good
                if not good:functional_failures.append([key,'consumer_check',seg['segment']])
                if condition=='chained_reencode':
                    base=seg['full_vs_cached'];assert math.isfinite(base['max'])
                    group=f'{model}|{dtype}|{shape}';full_errors[group]=max(full_errors[group],base['max'])
            negatives=row['rejects'];counts['rejection_observations']+=len(negatives)
            expected_rejects=({name for name in wanted if name in fields}|{f'hop{i}_epoch_only' for i in range(3)}|{'ciphertext_byte','truncation'}) if condition=='chained_reencode' else ({name for name in wanted if name in fields}|({'ciphertext_byte','truncation'} if condition!='shared_prefix' else set()))
            if {r['test'] for r in negatives}!=expected_rejects or len(negatives)!=len(expected_rejects):issues.append([key,'reject_set'])
            for negative in negatives:
                test=negative['test'];field='epoch_id' if test.startswith('hop') else fields.get(test)
                if negative.get('rejected') is not True:functional_failures.append([key,'rejection_failed',test])
                if check.get(test)!=negative.get('rejected'):issues.append([key,'rejection_flag_disagrees',test])
                counts['rejections_recorded']+=negative.get('rejected') is True
                if field:
                    if negative['changed_context_fields']!=[field] or negative['artifact_changed']:issues.append([key,'confounded_context_test',test])
                    if field=='principal' and negative.get('error')!='MAC verification failed':issues.append([key,'principal_rejection_cause',negative.get('error')])
                elif not negative['artifact_changed'] or negative['changed_context_fields']:issues.append([key,'artifact_test_misbound',test])
        if require_complete and file.stat().st_size!=start_size:issues.append([str(file),'changed_during_final_audit'])
    fixture_count=0
    original=root.parent/'defense_handoff_v2_20261031_1319/fixtures'
    overlapping=0
    for model in models:
        for sid in ids:
            variants=read(root/'fixtures'/model/(sid+'.json'))
            assert set(variants)==set(shapes)
            for shape,fix in variants.items():
                assert fix['sid']==sid and fix['model']==model and fix['shape']==shape
                assert len(fix['ids'])==len(fix['mask'])==len(fix['pos'])==fix['hist_len']
                assert len(fix['cont'])==fix['cont_len']==6 and len(fix['prefix_ids'])==16
                assert fix['ids'][:16]==fix['prefix_ids'] and set(fix['mask'])<={0,1}
                old=original/f'base_{model}_{shape}_{sid}.json'
                if old.exists():
                    prior=read(old)
                    for name in ('ids','mask','pos','cont','hist_len','cont_len','prefix_ids'):assert prior[name]==fix[name]
                    overlapping+=1
                fixture_count+=1
    assert fixture_count==6020 and overlapping==1000
    for env in (root/'full').glob('*/environment-*.json'):
        data=read(env)
        assert data['model_revision'] in {'060db6499f32faf8b98477b0a26969ef7d8b9987','9535bd9b1d1dea6acafbdc4813b728796aeb28da'}
        assert data['threads']==4 and data['dtype'] in dtypes
    project=Path(__file__).resolve().parents[1]
    source=root/'source/defense_runtime';reference=project/'github-public-artifact/defense_runtime'
    if not reference.exists():reference=project/'defense_runtime'
    for path in source.rglob('*.py'):
        relative=path.relative_to(source)
        if str(relative)=='corrected/freshness.py':continue
        source_checks[str(relative)]=path.read_bytes()==(reference/relative).read_bytes()
    assert all(source_checks.values())
    ledger=read(root/'ledger_stress/summary.json');assert ledger['pass'] and ledger['rounds_passed']==100 and ledger['concurrent_operations']==3200
    restart=read(root/'ledger_stress/PROCESS_RESTART_CHECK.json')
    overlap=read(root/'ledger_stress/OVERLAP_CHECK.json')
    assert restart['passed'] and restart['rejected']==100
    assert overlap['rounds_with_overlapping_claims']==100 and overlap['peak_inflight_min']==8
    fresh_environment=read(root/'FRESH_ENV_RUNTIME_CHECK.json')
    assert fresh_environment['all_pass'] and fresh_environment['unit_cases']==16
    resource_control={}
    if (root/'COMPLETED_BEFORE_RESCHEDULE.json').exists():
        snapshot=read(root/'COMPLETED_BEFORE_RESCHEDULE.json')
        retained=set(snapshot['keys'])
        assert len(retained)==len(snapshot['keys'])==5024
        assert all('|Llama-3.2-1B|float32|' in key for key in retained)
        assert retained<=seen
        retired=[]
        for path in (root/'full').glob('*/RETIRED.json'):
            info=read(path);count=sum(1 for line in (path.parent/'checks.jsonl').open() if line.strip())
            assert count==info['complete_records_retained'];retired.append(info)
        assert len(retired)==8 and sum(row['complete_records_retained'] for row in retired)==5024
        resource_control={'completed_records_retained_before_reshard':len(retained),'retired_workers':8,'remaining_requests_reassigned':12040-len(retained),'duplicate_complete_keys':counts['records']-len(seen),'reschedule_measurement_helper_source':read(root/'RESOURCE_RESCHEDULE_SOURCE_CHECK.json')}
        if (root/'full/EXIT_CODES.json').exists():resource_control['initial_worker_exit_codes']=read(root/'full/EXIT_CODES.json')
        if (root/'full/RESCHEDULE_EXIT_CODES.json').exists():resource_control['resumed_worker_exit_codes']=read(root/'full/RESCHEDULE_EXIT_CODES.json')
    complete=seen==expected and counts['records']==len(expected)
    if require_complete and not complete:issues.append(['incomplete',len(expected-seen)])
    result={'complete':complete,'record_integrity_pass':not issues and complete,'measurement_protocol_accepted':not issues and complete,'accepted_for_paper':not issues and complete,
            'expected_requests':len(expected),'counts':dict(counts),'missing_keys_count':len(expected-seen),'issues':issues,'functional_failures':functional_failures,'cells':{k:dict(v) for k,v in cells.items()},
            'shape_fixtures':fixture_count,'overlapping_old_fixtures_equal':overlapping,'baseline_full_vs_cached_maxima':dict(full_errors),'unchanged_runtime_source_except_freshness_resource_fix':source_checks,'ledger':ledger,'ledger_process_restart':restart,'ledger_actual_overlap':overlap,'fresh_environment_unit_checks':fresh_environment,
            'resource_control':resource_control,'acceptance_scope':'Known-key CPU software byte-cache functions on the frozen OR602 cohort. Does not establish a real serving scheduler, GPU, hardware trust boundary or within-request attack security.'}
    destination=root/('CODEX_VALIDATION.json' if require_complete else 'PARTIAL_VALIDATION.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','baseline_full_vs_cached_maxima','unchanged_runtime_source_except_freshness_resource_fix')},indent=2))
    if issues or require_complete and not complete:raise SystemExit('Independent data audit requires attention.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--complete',action='store_true');args=parser.parse_args();audit(args.root,args.complete)
