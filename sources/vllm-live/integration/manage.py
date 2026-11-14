"""Status, transport repair and private result handoff; never changes a score."""
import argparse,fcntl,json,re,time,zipfile
from pathlib import Path
from . import ROOT


def output_root():
    value=Path(json.loads((ROOT/'runtime/settings.json').read_text())['output_dir']);return value if value.is_absolute() else ROOT/value


def status():
    out=output_root();reports=[]
    if not out.exists():return []
    for folder in sorted(out.iterdir()):
        if not folder.is_dir():continue
        name=folder.name
        if not re.fullmatch(r'(smoke_)?(defense|attack)_(qwen25_05b|llama32_1b)_b(16|32)',name):continue
        path=folder/'records.jsonl';rows=[]
        if path.exists():
            with path.open('rb') as f:
                for line in f:
                    if line.endswith(b'\n'):rows.append(json.loads(line))
        expected=(2 if name.startswith('smoke_') else 602) if 'attack_' in name else (12 if name.startswith('smoke_') else 6321)
        reports.append(dict(job=name,measured=len(rows),expected=expected,complete=(folder/'summary.json').exists(),seconds_since_last_record=round(time.time()-rows[-1]['recorded_unix']) if rows else None,attempt_records=sum(1 for _ in (folder/'attempts.jsonl').open()) if (folder/'attempts.jsonl').exists() else 0))
    return reports


def repair(job):
    assert re.fullmatch(r'(smoke_)?(defense|attack)_(qwen25_05b|llama32_1b)_b(16|32)',job)
    folder=output_root()/job;lock=(folder/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=folder/'records.jsonl';data=p.read_bytes()
    if data.endswith(b'\n'):
        for x in data.splitlines():json.loads(x)
        return dict(repaired=False,reason='All rows complete')
    cut=data.rfind(b'\n')+1
    for x in data[:cut].splitlines():json.loads(x)
    dest=folder/'transport_fragments';dest.mkdir(exist_ok=True);old=dest/('original-'+str(time.time_ns())+'.jsonl.partial');p.rename(old);p.write_bytes(data[:cut])
    return dict(repaired=True,original_bytes_preserved=str(old.relative_to(folder)),all_complete_rows_unchanged=True,resume_same_unfinished_id=True)


def export():
    out=output_root();dest=ROOT/'vllm-live-private-handoff.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix!='.lock':z.write(p,'results/'+p.relative_to(out).as_posix())
        for name in ['SCIENTIFIC_LOCK.json','SCIENTIFIC_BASELINE.zip','runtime/settings.json']:
            z.write(ROOT/name,name)
        z.writestr('HANDOFF_SCOPE.txt','Private acceptance handoff: original observations, errors, runtime metadata and ledger state. Normalize portable accepted copies before anonymous publication.\n')
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    return dict(archive=str(dest),bytes=dest.stat().st_size,private_handoff=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','export','repair']);p.add_argument('--job');a=p.parse_args();result=status() if a.action=='status' else export() if a.action=='export' else repair(a.job);print(json.dumps(result,indent=2))
