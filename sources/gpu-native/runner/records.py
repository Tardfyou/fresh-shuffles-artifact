"""Append-only observations and strict completion accounting."""
from pathlib import Path
import json
import os
import time
import uuid
import fcntl

CURRENT_INVOCATION=None

class PauseRequested(BaseException):
    pass


def load(path):
    if not Path(path).exists():
        return []
    rows=[]
    with Path(path).open() as f:
        for n,line in enumerate(f,1):
            if not line.endswith('\n'):
                raise ValueError(f'Incomplete trailing record {path}:{n}; preserve and repair its transport before resume')
            rows.append(json.loads(line))
    return rows


class Records:
    def __init__(self, root, job):
        global CURRENT_INVOCATION
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.writer_lock=(self.root/'JOB.lock').open('a')
        try:fcntl.flock(self.writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.writer_lock.close()
            raise RuntimeError('Another worker owns this job output directory')
        self.path=self.root/'records.jsonl'
        self.job=job
        self.invocation=uuid.uuid4().hex
        lock=Path(__file__).resolve().parents[1]/'SCIENTIFIC_LOCK.json'
        self.science_version=json.loads(lock.read_text())['version'] if lock.exists() else 'development_cpu_validation'
        CURRENT_INVOCATION=self.invocation
        self.done={}
        for row in load(self.path):
            assert row['job_id']==job['id']
            if row['status']=='measured':
                assert row['sid'] not in self.done, 'Duplicate measured observation'
                self.done[row['sid']]=row

    def append(self,sid,**fields):
        assert sid not in self.done, 'Never retry a normally completed negative outcome'
        row=dict(job_id=self.job['id'],sid=sid,invocation_id=self.invocation,science_version=self.science_version,recorded_unix=time.time(),**fields)
        with self.path.open('a') as f:
            f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
        if row['status']=='measured':self.done[sid]=row
        if (Path(__file__).resolve().parents[1]/'runtime/pause.flag').exists():
            raise PauseRequested('Paused after preserving a complete observation')

    def finish(self, expected):
        assert set(self.done)==set(expected), f'Incomplete IDs: {len(self.done)}/{len(expected)}'
        summary=dict(job_id=self.job['id'],complete=True,expected=len(expected),measured=len(self.done),
                     errors_retained=sum(r['status']=='error' for r in load(self.path)),normal_negative_outcomes_retained=True)
        (self.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary

    def close(self):
        self.writer_lock.close()
