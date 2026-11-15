"""Operate the standalone length-stratified latency supplement."""
import argparse,json,os,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def absolute(base,value):
 p=Path(value).expanduser();return Path(os.path.abspath(p if p.is_absolute() else base/p))
def configured():return read(ROOT/'runtime/CONFIGURED.json')
def command(python,*args):
 subprocess.run([str(python),'-u',*args],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false'),check=True)
def configure(a):
 old=Path(a.old_native).expanduser().resolve()
 if not (old/'runtime/settings.json').exists() and (old/'native/runtime/settings.json').exists():old=old/'native'
 s=read(old/'runtime/settings.json');python=absolute(old,a.python or s['environment_python']['native']);cache=absolute(old,a.cache_dir or s['cache_dir']);assert python.is_file() and cache.is_dir()
 settings=read(ROOT/'runtime/settings.json');settings.update(environment_python={'native':str(python)},cache_dir=str(cache),gpu_ids=[a.gpu],output_dir='results',cpu_threads_per_job=a.threads)
 write(ROOT/'runtime/settings.json',settings);write(ROOT/'runtime/CONFIGURED.json',dict(native_python=str(python),cache_dir=str(cache),gpu=a.gpu,threads=a.threads,original_package=str(old)));print('Configured existing native environment; no model is run')
def run(a):
 python=configured()['native_python'];retry=['--retry-blocked'] if a.retry_blocked else []
 command(python,'-c','from runner.lock import verify; print(verify())')
 if not (ROOT/'results/SMOKE_GATE.json').exists():command(python,'-m','runner.supervisor','--phase','smoke',*retry)
 if not a.smoke_only:command(python,'-m','runner.supervisor','--phase','performance',*retry)
def status():
 from runner.records import load
 jobs=[]
 for j in read(ROOT/'protocol/jobs.json'):
  rows=load(ROOT/'results'/j['id']/'records.jsonl');n=sum(x['status']=='measured' for x in rows)
  jobs.append(dict(job=j['id'],measured=n,expected=j['expected_observations'],complete=(ROOT/'results'/j['id']/'summary.json').exists()))
 print(json.dumps(dict(jobs=jobs,measured=sum(j['measured'] for j in jobs),expected=180000),indent=2))
def verify():
 python=configured()['native_python']
 command(python,'-c','from runner.lock import verify; print(verify())')
 for file in ['smoke_jobs.json','jobs.json']:
  for j in read(ROOT/'protocol'/file):assert (ROOT/'results'/j['id']/'summary.json').exists(),'Incomplete job: '+j['id']
  command(python,'-m','runner.verify_outputs','--manifest',str(ROOT/'protocol'/file),'--output',str(ROOT/'results'))
 command(python,'-m','runner.latency_summary','--output',str(ROOT/'results'))
 write(ROOT/'runtime/VERIFIED.json',dict(complete=True,formal_records=180000,smoke_records=360,source_version=read(ROOT/'SCIENTIFIC_LOCK.json')['version']))
def export(partial):
 if not partial:assert (ROOT/'runtime/VERIFIED.json').exists(),'Run verify first, or export --partial for an incomplete run'
 target=ROOT/('latency-lengths-private-handoff-'+time.strftime('%Y%m%d-%H%M%S')+'.zip');assert not target.exists()
 manifest=read(ROOT/'SCIENTIFIC_LOCK.json');missing=[name for name in manifest['files'] if not (ROOT/name).is_file()];assert not missing,missing
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name in ['SCIENTIFIC_LOCK.json','SCIENTIFIC_BASELINE.zip']:z.write(ROOT/name,name)
  for name in manifest['files']:z.write(ROOT/name,'source_snapshot/'+name)
  for part in ['results','runtime']:
   for p in sorted((ROOT/part).rglob('*')):
    if p.is_file() and p.suffix!='.lock' and p.name!='JOB.lock':z.write(p,p.relative_to(ROOT).as_posix())
  z.writestr('RETURN_SCOPE.json',json.dumps(dict(partial=partial,actual_source_snapshot_included=True,scientific_baseline_included=True,source_files=len(manifest['files']),all_logs_and_errors_retained=True),indent=2))
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 print(json.dumps(dict(private_handoff=str(target),bytes=target.stat().st_size)))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 a=sub.add_parser('configure');a.add_argument('--old-native',required=True);a.add_argument('--python');a.add_argument('--cache-dir');a.add_argument('--gpu',type=int,default=0);a.add_argument('--threads',type=int,default=16,choices=range(1,33))
 a=sub.add_parser('run');a.add_argument('--smoke-only',action='store_true');a.add_argument('--retry-blocked',action='store_true')
 for name in ['status','verify']:sub.add_parser(name)
 a=sub.add_parser('export');a.add_argument('--partial',action='store_true');args=p.parse_args()
 {'configure':lambda:configure(args),'run':lambda:run(args),'status':status,'verify':verify,'export':lambda:export(args.partial)}[args.action]()
