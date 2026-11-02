"""Keep this project's throughput jobs out of its CPU timing measurements."""
import json
import os
from pathlib import Path
import time


def wait_for_own_jobs(output):
    here=Path(__file__).resolve().parent
    heavy={'order.py','direction.py','cross_key.py','suite.py','probe_keys.py',
           'compare_calibrators.py','privacy.py','privacy_controls.py','spectrum.py','integrity.py',
           'launch_order.py','launch_cross_key.py','launch_suite.py','launch_privacy_controls.py',
           'supervise.py'}
    heavy.add('reproduce_one.py')
    tool_heavy={'defense_validation_suite.py','review_defense_cache_semantics.py'}
    tools_directory=here.parents[2]/'tools'
    announced=False
    while True:
        active=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                if proc.stat().st_uid!=os.getuid():continue
                if (proc/'stat').read_text().split(') ',1)[1].startswith('Z'):continue
                cwd=(proc/'cwd').resolve()
                for arg in (proc/'cmdline').read_bytes().decode(errors='replace').split('\0'):
                    path=Path(arg)
                    if path.name not in heavy and path.name not in tool_heavy:continue
                    path=path if path.is_absolute() else cwd/path
                    if (path.name in heavy and path.parent.resolve()==here) or (path.name in tool_heavy and path.parent.resolve()==tools_directory):
                        active.append(path.name);break
            except (OSError,IndexError):continue
        if not active:break
        if not announced:
            print('Performance measurement queued until other local experiment workers exit.',flush=True)
            announced=True
        time.sleep(30)
    context=dict(own_experiment_workers=0,system_load_average=list(os.getloadavg()),
                 logical_cpus=os.cpu_count(),torch_threads=int(os.environ.get('VEILKV_THREADS','4')),
                 scope='Other users and system services are outside this project; shared-host load is recorded')
    path=Path(output)/'performance_context.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(context,indent=2))
    return context
