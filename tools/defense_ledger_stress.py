"""Known-key local freshness-ledger concurrency and resource-lifetime checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time


def operation(args):
    root,path,kind,scope,payload,start=args
    sys.path.insert(0,str(Path(root)/'source/defense_runtime'))
    from corrected.freshness import KeyService,FreshnessError
    service=KeyService(path)
    time.sleep(max(0,start-time.monotonic()))
    began=time.monotonic_ns()
    try:
        if kind=='claim':service.claim(scope)
        else:service.public(scope,payload)
        status='accepted';error=None
    except FreshnessError as exc:status='rejected';error=str(exc)
    return {'status':status,'error':error,'started_ns':began,'ended_ns':time.monotonic_ns()}


def run(root,rounds=100,width=8):
    sys.path.insert(0,str(root/'source/defense_runtime'))
    from corrected.freshness import KeyService,FreshnessError,scope
    output=root/'ledger_stress';output.mkdir(exist_ok=True)
    path=output/'freshness.sqlite';service=KeyService(path)
    records=[]
    def parallel(pool,kind,keys,payloads):
        start=time.monotonic()+0.05
        return list(pool.map(operation,[(str(root),str(path),kind,k,p,start) for k,p in zip(keys,payloads)]))
    with ProcessPoolExecutor(max_workers=width) as pool:
        for round_id in range(rounds):
            same=scope('same',round_id)
            collision=parallel(pool,'claim',[same]*width,[b'']*width)
            distinct=parallel(pool,'claim',[scope('distinct',round_id,i) for i in range(width)],[b'']*width)
            public=scope('public',round_id)
            consistent=parallel(pool,'public',[public]*width,[b'public-bytes']*width)
            conflict=parallel(pool,'public',[public]*width,[b'other-bytes']*width)
            restarted=KeyService(path)
            try:restarted.claim(same);restart_rejected=False
            except FreshnessError:restart_rejected=True
            check={'same_nonce_exactly_one':sum(r['status']=='accepted' for r in collision)==1,
                   'distinct_nonces_all':all(r['status']=='accepted' for r in distinct),
                   'public_idempotent_all':all(r['status']=='accepted' for r in consistent),
                   'public_conflicts_all_rejected':all(r['status']=='rejected' for r in conflict),
                   'restart_replay_rejected':restart_rejected}
            records.append({'round':round_id,'checks':check,'same_nonce':collision,'distinct':distinct,'public_same':consistent,'public_conflict':conflict,'pass':all(check.values())})
            if (round_id+1)%10==0:print(json.dumps({'ledger_rounds':round_id+1,'passed':sum(r['pass'] for r in records)}),flush=True)
    rollback=scope('rollback')
    try:
        with service.transaction() as connection:
            connection.execute('INSERT INTO nonces VALUES (?)',(rollback,))
            raise RuntimeError('deliberate rollback control')
    except RuntimeError:pass
    service.claim(rollback)
    before=len(list(Path('/proc/self/fd').iterdir()))
    for i in range(1000):service.claim(scope('resource',i))
    after=len(list(Path('/proc/self/fd').iterdir()))
    summary={'rounds':rounds,'concurrent_workers':width,'concurrent_operations':rounds*width*4,
             'rounds_passed':sum(r['pass'] for r in records),'rollback_control_passed':True,
             'handle_count_before':before,'handle_count_after':after,'resource_calls':1000,
             'pass':all(r['pass'] for r in records) and after<=before+1,
             'scope':'Software SQLite state only; no hardware anti-rollback or remote-service claim.'}
    (output/'checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not summary['pass']:raise SystemExit('Ledger stress requires investigation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--rounds',type=int,default=100);parser.add_argument('--width',type=int,default=8)
    args=parser.parse_args();run(args.root,args.rounds,args.width)
