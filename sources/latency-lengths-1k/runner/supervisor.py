"""Resource-limited queue: one 80-GiB card by default, resumable by observation."""
import argparse
import csv
import fcntl
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import ROOT
from .lock import verify
from .verify_outputs import verify as verify_job


def gpu_inventory():
    cmd=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=10)
    gpus=[]
    for r in csv.reader(p.stdout.splitlines()):
        index,uid,name,total,free,util=[v.strip() for v in r]
        gpus.append(dict(index=int(index),uuid=uid,name=name,total_gib=float(total)/1024,free_gib=float(free)/1024,utilization=float(util)))
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    for gpu in gpus:gpu['compute_pids']=[]
    for line in p.stdout.splitlines():
        if ',' not in line:continue
        uid,pid=[s.strip() for s in line.split(',',1)]
        if pid.isdigit():
            for gpu in gpus:
                if gpu['uuid']==uid:gpu['compute_pids'].append(int(pid))
    return gpus


def child_pids(pid):
    answer={pid};todo=[pid]
    while todo:
        current=todo.pop();path=Path(f'/proc/{current}/task/{current}/children')
        try:children=[int(p) for p in path.read_text().split()]
        except OSError:children=[]
        for child in children:
            if child not in answer:answer.add(child);todo.append(child)
    return answer


def settings(path):
    s=json.loads(path.read_text())
    assert 1<=s['max_concurrent_jobs']<=2,'Single-card package allows at most two small resident jobs'
    assert len(s['gpu_ids'])==1,'This package is sized for one GPU'
    assert 1<=s['cpu_threads_per_job']<=32
    assert s['reserved_gpu_gib']>=2
    assert 1<=s['maximum_gpu_allocation_gib']<=78
    return s


def count_records(path):
    if not path.exists():return 0
    n=0
    with path.open() as f:
        for line in f:
            if line.endswith('\n'):
                try:n+=json.loads(line).get('status')=='measured'
                except json.JSONDecodeError:pass
    return n


class ExistingWorker:
    """Adopt an owned worker after its previous monitor exited."""
    def __init__(self,pid,folder):self.pid=pid;self.folder=folder
    def poll(self):
        path=Path(f'/proc/{self.pid}/stat')
        try:state=path.read_text().rpartition(') ')[2].split()[0]
        except OSError:state=None
        if state and state!='Z':return None
        return 0 if (self.folder/'summary.json').exists() else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','core','performance','engine_baseline','official_utility','all'],default='core');p.add_argument('--settings',type=Path,default=ROOT/'runtime/settings.json');p.add_argument('--retry-blocked',action='store_true');a=p.parse_args()
    verify()
    s=settings(a.settings);output=Path(s['output_dir']);output=output if output.is_absolute() else ROOT/output;output.mkdir(parents=True,exist_ok=True)
    monitor_lock=(output/'SUPERVISOR.lock').open('a')
    try:fcntl.flock(monitor_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A supervisor is already monitoring this output directory')
    manifest=ROOT/'protocol'/('smoke_jobs.json' if a.phase=='smoke' else 'jobs.json')
    jobs=json.loads(manifest.read_text())
    if a.phase not in ['smoke','all']:jobs=[j for j in jobs if j['phase']==a.phase]
    jobs=sorted(jobs,key=lambda j:j.get('priority',0))
    gate=output/'SMOKE_GATE.json'
    if a.phase!='smoke':
        assert gate.exists() and json.loads(gate.read_text())['passed'],'Run and accept the unchanged smoke suite first'
    statefile=output/('QUEUE_'+a.phase+'.json');states=json.loads(statefile.read_text()) if statefile.exists() else {}
    # A restarted monitor must not launch over an orphan from another phase.
    for oldfile in output.glob('QUEUE_*.json'):
        if oldfile==statefile:continue
        for jid,entry in json.loads(oldfile.read_text()).items():
            if entry.get('status')!='running':continue
            pid=entry.get('pid');cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists():
                command=cmd.read_bytes().decode(errors='replace').split('\0')
                if 'runner.worker' in command and jid in command and str(output) in command:
                    raise RuntimeError('An owned worker from '+oldfile.name+' is still active. Resume that phase before starting a different phase.')
    running={}
    for job in jobs:
        entry=states.get(job['id'],{})
        if entry.get('status')=='running':
            pid=entry.get('pid');cmdfile=Path(f'/proc/{pid}/cmdline')
            if cmdfile.exists():
                command=cmdfile.read_bytes().decode(errors='replace').split('\0')
                if 'runner.worker' in command and job['id'] in command:
                    running[job['id']]=dict(process=ExistingWorker(pid,output/job['id']),job=job,gpu=entry.get('gpu'),log=open(os.devnull,'w'))
                elif command:entry.update(status='blocked',reason='Saved PID belongs to a different command; do not start a duplicate')
                else:entry={'status':'pending','previous_pid':pid}
            else:entry={'status':'pending','previous_pid':pid}
        if entry.get('status')=='blocked' and a.retry_blocked:entry['status']='pending'
        states[job['id']]=entry or {'status':'pending'}
    last_print=0
    try:
        while True:
            s=settings(a.settings)
            for jid,item in list(running.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();job=item['job']
                if code==0:
                    try:
                        verify();validation=verify_job(job,output/jid)
                        if a.phase=='smoke' and job['kind'] in ['byte_defense','large_byte']:
                            assert validation['normal_negative_outcomes']==0,'Byte round-trip smoke failed; retain it and request code review'
                        states[jid].update(status='complete',validation=validation,finished_unix=time.time())
                    except Exception as e:states[jid].update(status='blocked',reason='Acceptance: '+str(e),finished_unix=time.time())
                elif code==75:states[jid].update(status='paused',finished_unix=time.time())
                else:states[jid].update(status='blocked',reason='Worker exited '+str(code)+'; retain logs and all records')
                del running[jid]
            paused=(ROOT/'runtime/pause.flag').exists()
            if not paused and len(running)<s['max_concurrent_jobs']:
                inventory=gpu_inventory()
                owned=set().union(*(child_pids(item['process'].pid) for item in running.values())) if running else set()
                for job in jobs:
                    if states[job['id']]['status'] not in ['pending','paused']:continue
                    if any(x['job'].get('exclusive') for x in running.values()) or (job.get('exclusive') and running):continue
                    python=ROOT/s['environment_python'][job.get('environment','native')]
                    if not python.exists():
                        states[job['id']].update(status='blocked',reason='Missing declared environment '+str(python.relative_to(ROOT)))
                        continue
                    occupied_by_queue={item.get('gpu') for item in running.values()}
                    eligible=[g for g in inventory if g['index'] in s['gpu_ids'] and g['free_gib']>=job['minimum_free_gib'] and (g['utilization']<=s['idle_utilization_threshold'] or g['index'] in occupied_by_queue) and (s.get('allow_external_gpu_processes',False) and not job.get('exclusive') or not(set(g['compute_pids'])-owned))]
                    if not eligible:continue
                    gpu=eligible[0];jid=job['id'];folder=output/jid;folder.mkdir(exist_ok=True)
                    if (folder/'summary.json').exists():
                        try:states[jid].update(status='complete',validation=verify_job(job,folder));continue
                        except Exception as e:states[jid].update(status='blocked',reason='Existing observations: '+str(e));continue
                    logpath=folder/('stdout-'+str(time.time_ns())+'.log');log=logpath.open('w')
                    cmd=[str(python),'-u','-m','runner.worker','--job',jid,'--manifest',str(manifest),'--output',str(output),'--device','cuda:0','--cache-dir',s['cache_dir'],'--threads',str(s['cpu_threads_per_job']),'--max-gpu-allocation-gib',str(s['maximum_gpu_allocation_gib'])]
                    if s.get('offline',False):cmd.append('--offline')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),PYTHONPATH=str(ROOT),TOKENIZERS_PARALLELISM='false',NVIDIA_TF32_OVERRIDE='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    states[jid].update(status='running',pid=proc.pid,gpu=gpu['index'],started_unix=time.time(),log=str(logpath.relative_to(output)))
                    running[jid]=dict(process=proc,job=job,gpu=gpu['index'],log=log)
                    print('START',jid,'GPU',gpu['index'],'PID',proc.pid,flush=True)
                    break  # Refresh actual free memory before admitting another job.
            for jid in running:states[jid]['measured_observations']=count_records(output/jid/'records.jsonl')
            statefile.write_text(json.dumps(states,indent=2)+'\n')
            if time.monotonic()-last_print>=30:
                counts={k:sum(v.get('status')==k for v in states.values()) for k in ['complete','running','pending','paused','blocked']}
                print(json.dumps(dict(phase=a.phase,**counts)),flush=True);last_print=time.monotonic()
            remaining=[j for j in jobs if states[j['id']]['status'] in ['pending','running','paused']]
            if not remaining or paused and not running:break
            time.sleep(min(30,max(2,s['poll_seconds'])))
    finally:
        statefile.write_text(json.dumps(states,indent=2)+'\n')
    complete=all(states[j['id']]['status']=='complete' for j in jobs)
    if a.phase=='smoke' and complete:gate.write_text(json.dumps(dict(passed=True,jobs=len(jobs),source_lock=verify()),indent=2)+'\n')
    if not complete and any(v['status']=='blocked' for v in states.values()):raise SystemExit(2)
    print('QUEUE',a.phase,'complete' if complete else 'paused',flush=True)


if __name__=='__main__':main()
