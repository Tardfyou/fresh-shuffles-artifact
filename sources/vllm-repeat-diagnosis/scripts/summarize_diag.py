"""Independent summary of diagnostic stage outputs. No engine required."""
import json, sys, glob, os

REC = sys.argv[1] if len(sys.argv) > 1 else 'records'

def load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(REC, pattern))):
        out.append((p, json.load(open(p))))
    return out

print('=== Stage1 contrast matrix ===')
print(f"{'variant':8s} {'block':5s} {'equal':>6s} {'orig-divergent still divergent':>32s} {'orig-consistent turned divergent':>33s}")
for p, d in load('stage1_*_b*.json'):
    v, b = d['variant'], d['block']
    rows = d['samples']
    eq = sum(1 for r in rows if r['equal'])
    still = sum(1 for r in rows if not r['equal'] and r['original_divergent'])
    turned = sum(1 for r in rows if not r['equal'] and not r['original_divergent'])
    print(f'{v:8s} b{b:<4d} {eq:3d}/{len(rows):<3d} {still:32d} {turned:33d}')

print('\n=== First-divergence numerics (divergent pairs in C/E; logprobs top-2) ===')
for p, d in load('stage1_C_b*.json') + load('stage1_E_b*.json') + load('stage1_B_b*.json') + load('stage1_A_b*.json') + load('stage1_D_b*.json'):
    for r in d['samples']:
        if r['equal'] or not r['a'].get('logprobs'):
            continue
        fd = r['first_div']
        # compare distribution at the divergence step of run a (history identical up to fd-1)
        for name, run in (('a', r['a']), ('b', r['b'])):
            lp = run['logprobs']
            if fd is not None and fd < len(lp) and lp[fd]:
                top = lp[fd][:2]
                gap = top[0][1] - top[1][1] if len(top) > 1 else None
                print(f"  {d['variant']} b{d['block']} {r['sid']} run={name} div@{fd} top1={top[0][0]}({top[0][1]:.4f}) "
                      f"top2={top[1][0] if len(top)>1 else None}({top[1][1]:.4f}) gap={gap:.2e}" if len(top) > 1 else
                      f"  {d['variant']} b{d['block']} {r['sid']} run={name} div@{fd} top1 only")
        # tie-swap check: is a's top2 == a's chosen(b's token)?
        lpa, tokb = r['a']['logprobs'], r['b']['token_ids']
        if lpa and fd is not None and fd < len(lpa) and lpa[fd] and len(lpa[fd]) > 1:
            a_top2 = lpa[fd][1][0]
            print(f"    tie-swap check {r['sid']}: a_top2={a_top2} b_token@fd={tokb[fd]} swap={a_top2 == tokb[fd]}")

print('\n=== C0 vs C instrumentation equivalence (same samples) ===')
c0 = {(d['block'], r['sid']): r for p, d in load('stage1_C0_b*.json') for r in d['samples']}
cc = {(d['block'], r['sid']): r for p, d in load('stage1_C_b*.json') for r in d['samples']}
common = set(c0) & set(cc)
mism = [(k, c0[k]['a']['token_ids'], cc[k]['a']['token_ids']) for k in common
        if c0[k]['a']['token_ids'] != cc[k]['a']['token_ids'] or c0[k]['b']['token_ids'] != cc[k]['b']['token_ids']]
print(f'pairs compared: {len(common)}; token mismatches with logprobs enabled: {len(mism)}')
for k, x, y in mism[:5]:
    print('  MISMATCH', k, x[:5], y[:5])
