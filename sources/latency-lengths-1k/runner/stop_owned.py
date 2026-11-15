"""Interrupt only this queue's verified worker PIDs; keep all raw records."""
import argparse
import json
from pathlib import Path
import os
import signal
from . import ROOT
from .paths import result_root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=result_root());a=p.parse_args()
    a.output=a.output.resolve()
    (ROOT/'runtime/pause.flag').touch()
    seen=set()
    for file in a.output.glob('QUEUE_*.json'):
        for jid,entry in json.loads(file.read_text()).items():
            pid=entry.get('pid')
            if not pid or pid in seen:continue
            commandfile=Path(f'/proc/{pid}/cmdline')
            try:command=commandfile.read_bytes().decode(errors='replace').split('\0')
            except OSError:continue
            if 'runner.worker' in command and jid in command and str(a.output) in command:
                os.kill(pid,signal.SIGINT);seen.add(pid);print('Interrupted owned worker',jid,pid)
    print('Raw observations are retained. Repair any incomplete final transport row before resuming.')
