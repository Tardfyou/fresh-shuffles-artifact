"""Check artifact navigation and GPU run references without running models."""
import argparse
import json
from pathlib import Path
import re


def read(path):return json.loads(path.read_text())


def check(root):
    catalog=read(root/'metadata/studies.json');ids=[];references=[]
    for study in catalog['studies']:
        ids.append(study['id']);references+=study['paths']
        if study.get('documentation'):references.append(study['documentation'])
    assert len(ids)==len(set(ids)), 'Duplicate study IDs'
    references += [catalog['fixed_manifest'],catalog['fixed_subsets'],*catalog['dataset_source_records']]
    references += [catalog['gpu'][k] for k in ['index','guide','expected_jobs']]
    for name in references:
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts, 'Catalog paths must be portable'
        assert (root/p).exists(), 'Missing catalog target: '+name
    layout=read(root/'metadata/layout.json')
    for old,new in layout['relocated_files'].items():
        assert (root/new).is_file() and not (root/old).exists(), 'Inconsistent relocated file: '+old
    expected=read(root/'metadata/gpu/protocol/jobs.json')
    assert len(expected)==len({j['id'] for j in expected})==93
    index=read(root/'results/gpu/index.json');run_ids=[]
    for run in index['runs']:
        run_ids.append(run['run_id'])
        assert run['status']=='accepted'
        for field in ['raw_path','summary_path','validation_path']:
            p=Path(run[field]);assert not p.is_absolute() and '..' not in p.parts
            assert (root/p).exists(), 'Missing accepted run reference'
        assert run['source_version'] and run['profiles']
    assert len(run_ids)==len(set(run_ids)), 'Duplicate GPU run IDs'
    for study in catalog['studies']:
        for job in study.get('primary_gpu_jobs', []):
            run=study.get('primary_gpu_job_runs', {}).get(job, study.get('primary_gpu_run'))
            assert run in run_ids, 'Missing study run: '+study['id']
            assert (root/'results/gpu/accepted'/run/job).is_dir(), 'Missing primary job: '+job
    for job,run in index.get('primary_job_overrides', {}).items():
        assert run in run_ids and (root/'results/gpu/accepted'/run/job).is_dir()
    assert index['status']=='accepted_gpu_primary_cpu_auxiliary'
    assert not index.get('pending_latency_lengths_1k', False)
    assert catalog['gpu']['latest_live602_run']==index['primary_live_run'] in run_ids
    assert catalog['gpu']['latest_latency_run']==index['primary_latency_run'] in run_ids
    assert layout['current_results_index']==catalog['gpu']['index']
    assert layout['execution_sources']==catalog['gpu']['execution_sources']
    for name in layout['execution_sources'].values():
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts and (root/p).is_dir()
    for key in ['original_main','native602','live602','latency_lengths_1k']:
        source=root/layout['execution_sources'][key]
        manifest=read(source/'SCIENTIFIC_LOCK.json')
        assert (source/manifest['baseline_archive']).is_file()
        plan='PLAN.json' if key=='live602' else 'jobs.json'
        assert (source/'protocol'/plan).is_file()
    original=root/'results/gpu/accepted/a100-80g-20261110-01'
    retired=read(root/'metadata/gpu602/RETIRED_COHORTS.json')
    for job in retired['retired_formal_jobs']:
        assert not (original/job).exists() and not (original/('smoke_'+job)).exists()
    for job in expected:
        if job['kind']=='latency':
            assert not (original/job['id']).exists()
            assert not (original/('smoke_'+job['id'])).exists()
    retired_live=read(root/'metadata/gpu602/RETIRED_LIVE_COHORT.json')
    assert retired_live['retired_run'] not in run_ids
    assert not (root/'results/gpu/accepted'/retired_live['retired_run']).exists()
    assert not (root/'results/gpu/summaries'/retired_live['retired_run']).exists()
    for document in [root/'README.md',*(root/'docs').glob('*.md')]:
        for link in re.findall(r'\[[^]]*\]\(([^)]+)\)',document.read_text()):
            if re.match(r'(?:https?|mailto):',link):continue
            target=link.split('#')[0]
            assert not target or (document.parent/target).exists(), 'Broken documentation link: '+link
    return dict(catalog_pass=True,studies=len(ids),checked_references=len(references),original_plan_execution_units=len(expected),accepted_gpu_runs=len(run_ids),current_run_and_source_navigation_pass=True,retired_gpu_cohorts_absent=True,documentation_links_pass=True,existing_scientific_paths_preserved=layout['existing_scientific_result_paths_preserved'],model_or_api_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(check(a.root),indent=2))
