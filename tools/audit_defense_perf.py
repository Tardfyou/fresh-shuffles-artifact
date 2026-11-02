"""Recompute the complete cache-operation timing matrix from retained samples."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def audit(root):
    result=json.loads((root/'result.json').read_text());protocol=result['protocol']
    operations={operation+'_'+name for operation in ('protect','recover') for name in ('tee','nosharing_aead','kvcloak','kvcloak_fused')}
    assert protocol['actual_tokens']==512 and protocol['threads']==4 and protocol['repetitions']==100
    assert protocol.get('input_sid') and protocol['input_source'].startswith('One existing frozen WildChat')
    assert all(row['typed_bytes_exact'] and row['values_exact'] for row in result['typed_roundtrip_checks'])
    seen=set();values={key:[] for key in operations}
    for line in (root/'timings.jsonl').open():
        row=json.loads(line);key=(row['round'],row['operation'])
        assert key not in seen and 0<=row['round']<100 and row['operation'] in operations
        seen.add(key);seconds=row['seconds'];normalized=row['seconds_per_1000_tokens']
        assert math.isfinite(seconds) and seconds>0 and abs(normalized-seconds*1000/512)<1e-12
        values[row['operation']].append(normalized)
    assert len(seen)==800 and seen=={(round_,op) for round_ in range(100) for op in operations}
    for key,samples in values.items():
        report=result['perf'][key]
        assert samples==report['samples_seconds_per_1000_tokens']
        for field,p in [('p50',.5),('p95',.95),('p99',.99)]:assert abs(report[field]-np.quantile(samples,p))<1e-12
        assert abs(report['mean']-np.mean(samples))<1e-12 and abs(report['std']-np.std(samples))<1e-12
    outcome={'raw_timing_cells':800,'operations':8,'repetitions_per_operation':100,'duplicate_cells':0,'all_statistics_recomputed':True,'input_sid':protocol['input_sid'],'typed_bytes_exact':True,'scope':'Known-key CPU cache operations, 512 real tokens, authenticated byte recovery plus trusted dtype restoration; no serving/inference throughput claim.'}
    (root/'PERFORMANCE_VALIDATION.json').write_text(json.dumps(outcome,indent=2)+'\n')
    return outcome


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();print(json.dumps(audit(args.root),indent=2))
