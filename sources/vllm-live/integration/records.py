"""Append complete observations once; retain infrastructure attempts separately."""
from pathlib import Path
import json,time,uuid,fcntl
from . import ROOT


class Records:
    def __init__(self,folder,job):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.job=job;self.invocation=uuid.uuid4().hex
        lockfile=ROOT/'SCIENTIFIC_LOCK.json'
        self.science_version=json.loads(lockfile.read_text())['version'] if lockfile.exists() else 'development_cpu_validation'
        self.lock=(self.folder/'writer.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.done={}
        path=self.folder/'records.jsonl'
        if path.exists():
            for line in path.open('rb'):
                assert line.endswith(b'\n'),'Preserve and repair an incomplete transport tail before resume'
                row=json.loads(line);assert row.get('science_version')==self.science_version, 'Use a fresh 602 result directory; do not mix historical 100-case records'
                assert row['key'] not in self.done;self.done[row['key']]=row
    def append(self,key,**data):
        assert key not in self.done,'Normally completed wrong outcomes must not be overwritten'
        row=dict(key=key,job=self.job,invocation=self.invocation,science_version=self.science_version,recorded_unix=time.time(),status='measured',**data)
        with (self.folder/'records.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');f.flush()
        self.done[key]=row
    def error(self,key,error,partial_data=None):
        import traceback
        with (self.folder/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(key=key,invocation=self.invocation,time=time.time(),error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),partial_data=partial_data))+'\n')
    def finish(self,keys):
        assert set(self.done)==set(keys),f'Incomplete accounting {len(self.done)}/{len(keys)}'
        report=dict(job=self.job,complete=True,expected=len(keys),measured=len(self.done),normal_negative_outcomes_retained=True)
        (self.folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report
    def close(self):self.lock.close()
