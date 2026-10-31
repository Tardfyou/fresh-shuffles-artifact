"""Recompute descriptive saved-record results without models or network access."""
import argparse
import gzip
import json
from pathlib import Path
import statistics


def rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def inspect(root):
    study=root/'results/saved_record_review_20261101'
    audit=json.loads((study/'SAVED_RECORD_AUDIT.json').read_text())
    answer={}
    for tag,heads in [('qwen25_05b',2),('llama32_1b',8)]:
        seen=set();correct=total=0;seconds=0.0
        historical={r['sid']:r for r in rows(root/'results/main_public'/tag/'per_sample.jsonl')}
        for row in rows(study/tag/'sample_costs.jsonl.gz'):
            assert row['sid'] not in seen
            seen.add(row['sid'])
            for seed,cell in row['seeds'].items():
                assert (cell['correct'],cell['total'])==(historical[row['sid']]['seeds'][seed]['correct'],historical[row['sid']]['seeds'][seed]['total'])
                correct+=cell['correct'];total+=cell['total'];seconds+=cell['decode_s']
        assert len(seen)==3000 and seen==set(historical)
        gaps=[];blocks=0
        for row in rows(study/tag/'failed_block_diagnostics.jsonl.gz'):
            block=row['block'];assert block['correct']<16;blocks+=1
            predicted={r['p'] for r in block['rows']}
            gaps.extend(g['gap'] for g in block['gold'] if g['tok'] not in predicted)
        expected=audit['attribution'][tag]
        assert (correct,total)==(expected['counts']['correct'],expected['counts']['total'])
        assert blocks==expected['counts']['miss_blocks']
        assert len(gaps)==expected['gaps']['absent_distinct_gold_in_miss_blocks']['n']
        assert statistics.median(gaps)==expected['gaps']['absent_distinct_gold_in_miss_blocks']['median']
        cost=seconds/(total/heads)*1000
        assert abs(cost-expected['decode_seconds_per_1000_target_token'])<1e-8
        answer[tag]={'samples':len(seen),'correct':correct,'head_token_opportunities':total,'failed_blocks':blocks,'decode_seconds_per_1000_target_token':cost,'absent_distinct_gold_gap_median':statistics.median(gaps)}
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();print(json.dumps(inspect(args.root),indent=2))
