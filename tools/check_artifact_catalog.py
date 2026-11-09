"""Check artifact navigation and GPU run references without running models."""
import argparse
import json
from pathlib import Path


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
    return dict(catalog_pass=True,studies=len(ids),checked_references=len(references),expected_gpu_execution_units=len(expected),accepted_gpu_runs=len(run_ids),existing_scientific_paths_preserved=layout['existing_scientific_result_paths_preserved'],model_or_api_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(check(a.root),indent=2))
