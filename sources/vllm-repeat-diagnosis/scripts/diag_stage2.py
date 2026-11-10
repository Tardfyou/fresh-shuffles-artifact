"""Stage 2: 3 rounds x {A, C} x {16, 32} x full 100 inputs, plain SamplingParams (mirrors original)."""
import json, os, sys, time

os.environ['VLLM_USE_V1'] = '0'
VARIANT = sys.argv[1]   # A|C
BLOCK = int(sys.argv[2])
ROUND = int(sys.argv[3])
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision('highest')
from vllm import LLM, SamplingParams

MODEL = 'model_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987'
BLOCKS = {16: 72, 32: 40}[BLOCK]
rows = [json.loads(l) for l in open('runtime/results/vllm_baseline_b16/records.jsonl') if l.endswith('\n')]
rows = [r for r in rows if r.get('status') == 'measured']
assert len(rows) == 100

engine = LLM(model=MODEL, tokenizer=MODEL, dtype='bfloat16', tensor_parallel_size=1,
             enforce_eager=True, enable_prefix_caching=(VARIANT == 'C'), block_size=BLOCK,
             max_model_len=1024, max_num_seqs=1, num_gpu_blocks_override=BLOCKS,
             gpu_memory_utilization=.5, distributed_executor_backend='uni')
sched = engine.llm_engine.scheduler[0]
CAP = sched.block_manager.get_num_free_gpu_blocks()
params = SamplingParams(temperature=0., max_tokens=16)

def gen(ids):
    t0 = time.perf_counter()
    o = engine.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)[0]
    return list(o.outputs[0].token_ids), time.perf_counter() - t0

results = []
for r in rows:
    sid, ids = r['sid'], r['input_ids']
    if VARIANT == 'C':
        ok = bool(sched.reset_prefix_cache())
        free = sched.block_manager.get_num_free_gpu_blocks()
        assert ok and free == CAP, 'reset failed'
    a, ta = gen(ids)
    b, tb = gen(ids)
    eq = a == b
    fd = None if eq else next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    results.append(dict(sid=sid, equal=eq, first_div=fd, a=a, b=b))
    if not eq:
        print(f'R{ROUND} {VARIANT} b{BLOCK} {sid} DIVERGE@{fd}', flush=True)

eq_n = sum(1 for x in results if x['equal'])
path = f'records/stage2_{VARIANT}_b{BLOCK}_r{ROUND}.json'
json.dump(dict(variant=VARIANT, block=BLOCK, round=ROUND, equal=eq_n, total=len(results),
               capacity=CAP, samples=results), open(path, 'w'), indent=1)
print(f'DONE stage2 {VARIANT} b{BLOCK} r{ROUND}: {eq_n}/100 -> {path}', flush=True)
