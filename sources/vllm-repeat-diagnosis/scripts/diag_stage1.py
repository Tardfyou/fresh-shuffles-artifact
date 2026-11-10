"""Stage 1 diagnostic: variants C0/C/B/D/A/E on 15 divergent + 5 consistent samples.

Mirrors runner/extended.py vllm_baseline engine config exactly except the studied knob.
One engine per variant; run as: diag_stage1.py <variant> <block_size> [output_dir]
"""
import json, math, os, sys, time

os.environ['VLLM_USE_V1'] = '0'
VARIANT = sys.argv[1]            # C0|C|B|D|A|E
BLOCK = int(sys.argv[2])         # 16|32
OUT = sys.argv[3] if len(sys.argv) > 3 else 'records'
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision('highest')
from vllm import LLM, SamplingParams

MODEL = 'model_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987'
BLOCKS = {16: 72, 32: 40}[BLOCK]
assert (MODEL.split('/')[-1] == '060db6499f32faf8b98477b0a26969ef7d8b9987')

# --- inputs: exact 100 input_ids from original b16 records (asserted equal to b32) ---
def load_inputs():
    base = 'runtime/results/'
    r16 = [json.loads(l) for l in open(base + 'vllm_baseline_b16/records.jsonl') if l.endswith('\n')]
    r32 = [json.loads(l) for l in open(base + 'vllm_baseline_b32/records.jsonl') if l.endswith('\n')]
    r16 = [r for r in r16 if r.get('status') == 'measured']
    r32 = [r for r in r32 if r.get('status') == 'measured']
    assert [r['sid'] for r in r16] == [r['sid'] for r in r32]
    assert all(a['input_ids'] == b['input_ids'] for a, b in zip(r16, r32))
    return r16

div = json.load(open('records/divergent_original.json'))
DIV_SID = {d['sid'] for d in div}
allrows = load_inputs()
CONSIST_SID = []
for r in allrows:
    if r['sid'] not in DIV_SID and len(CONSIST_SID) < 5:
        CONSIST_SID.append(r['sid'])
if VARIANT == 'E':
    rows = [r for r in allrows if r['sid'] in DIV_SID]
else:
    rows = [r for r in allrows if r['sid'] in DIV_SID or r['sid'] in CONSIST_SID]
print(f'variant={VARIANT} block={BLOCK} samples={len(rows)}', flush=True)

dtype = 'float32' if VARIANT == 'E' else 'bfloat16'
caching = VARIANT != 'A'
engine = LLM(model=MODEL, tokenizer=MODEL, dtype=dtype, tensor_parallel_size=1,
             enforce_eager=True, enable_prefix_caching=caching, block_size=BLOCK,
             max_model_len=1024, max_num_seqs=1, num_gpu_blocks_override=BLOCKS,
             gpu_memory_utilization=.5, distributed_executor_backend='uni')
sched = engine.llm_engine.scheduler[0]
CAPACITY = sched.block_manager.get_num_free_gpu_blocks()

def reset_cache(tag):
    ok = bool(sched.reset_prefix_cache())
    free = sched.block_manager.get_num_free_gpu_blocks()
    rec = dict(tag=tag, reset_return=ok, free_blocks_after=free, capacity=CAPACITY,
               reset_verified=(ok and free == CAPACITY))
    print('RESET', json.dumps(rec), flush=True)
    return rec

def gen(ids, want_logprobs):
    p = SamplingParams(temperature=0., max_tokens=16, logprobs=20 if want_logprobs else None)
    free_before = sched.block_manager.get_num_free_gpu_blocks()
    t0 = time.perf_counter()
    o = engine.generate([{'prompt_token_ids': ids}], p, use_tqdm=False)[0]
    dt = time.perf_counter() - t0
    free_after = sched.block_manager.get_num_free_gpu_blocks()
    m = o.metrics
    metrics = None if m is None else {k: getattr(m, k) for k in dir(m)
                                      if not k.startswith('_') and isinstance(getattr(m, k), (int, float))}
    lp = None
    if want_logprobs and o.outputs[0].logprobs:
        lp = []
        for pos, d in enumerate(o.outputs[0].logprobs):
            if d is None:
                lp.append(None)
            else:
                items = sorted(d.items(), key=lambda kv: -kv[1].logprob)[:20]
                lp.append([(int(tok), float(lg.logprob)) for tok, lg in items])
    return dict(token_ids=list(o.outputs[0].token_ids), finish=o.outputs[0].finish_reason,
                seconds=dt, free_before=free_before, free_after=free_after,
                metrics=metrics, logprobs=lp)

results = []
WANT_LP = VARIANT != 'C0'
for r in rows:
    sid, ids = r['sid'], r['input_ids']
    if VARIANT in ('B', 'C', 'E'):
        rs = reset_cache(sid + ':pair'); a = gen(ids, WANT_LP); b = gen(ids, WANT_LP)
    elif VARIANT == 'C0':
        rs = reset_cache(sid + ':pair'); a = gen(ids, False); b = gen(ids, False)
    elif VARIANT == 'D':
        rs = dict(tag=sid + ':prewarm', reset_return=None, free_blocks_after=None,
                  capacity=CAPACITY, reset_verified=None)
        gen(ids, False)  # prewarm, unmeasured
        a = gen(ids, WANT_LP); b = gen(ids, WANT_LP)
    else:  # A
        rs = None; a = gen(ids, WANT_LP); b = gen(ids, WANT_LP)
    equal = a['token_ids'] == b['token_ids']
    fd = None
    if not equal:
        fd = next((i for i, (x, y) in enumerate(zip(a['token_ids'], b['token_ids'])) if x != y),
                  min(len(a['token_ids']), len(b['token_ids'])))
    results.append(dict(sid=sid, in_len=len(ids), variant=VARIANT, block=BLOCK, dtype=dtype,
                        a=a, b=b, equal=equal, first_div=fd, reset=rs,
                        original_divergent=sid in DIV_SID))
    print(sid, 'equal' if equal else f'DIVERGE@{fd}', flush=True)

path = f'{OUT}/stage1_{VARIANT}_b{BLOCK}.json'
json.dump(dict(variant=VARIANT, block=BLOCK, dtype=dtype, caching=caching,
               capacity_blocks=CAPACITY, num_gpu_blocks_override=BLOCKS,
               model_revision='060db6499f32faf8b98477b0a26969ef7d8b9987',
               vllm=__import__('vllm').__version__, samples=results), open(path, 'w'), indent=1)
neq = sum(1 for x in results if not x['equal'])
print(f'DONE {VARIANT} b{BLOCK}: {len(results)-neq}/{len(results)} equal -> {path}', flush=True)
