"""Variant F: cold-cold (B structure) with prompt_logprobs=1 to compare prefill numerics."""
import json, os, sys, time

os.environ['VLLM_USE_V1'] = '0'
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 16
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision('highest')
from vllm import LLM, SamplingParams

MODEL = 'model_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987'
BLOCKS = {16: 72, 32: 40}[BLOCK]
div = json.load(open('records/divergent_original.json'))
DIV_SID = sorted({d['sid'] for d in div})
rows = [json.loads(l) for l in open('runtime/results/vllm_baseline_b16/records.jsonl') if l.endswith('\n')]
rows = [r for r in rows if r.get('status') == 'measured' and r['sid'] in DIV_SID]

engine = LLM(model=MODEL, tokenizer=MODEL, dtype='bfloat16', tensor_parallel_size=1,
             enforce_eager=True, enable_prefix_caching=True, block_size=BLOCK,
             max_model_len=1024, max_num_seqs=1, num_gpu_blocks_override=BLOCKS,
             gpu_memory_utilization=.5, distributed_executor_backend='uni')
sched = engine.llm_engine.scheduler[0]
CAP = sched.block_manager.get_num_free_gpu_blocks()

def gen_prompt_lp(ids):
    p = SamplingParams(temperature=0., max_tokens=16, logprobs=20, prompt_logprobs=1)
    o = engine.generate([{'prompt_token_ids': ids}], p, use_tqdm=False)[0]
    plp = o.prompt_logprobs  # list len==len(ids); [0] is None
    vals = []
    for pos, d in enumerate(plp):
        if d is None:
            vals.append(None)
        else:
            tok = ids[pos]
            vals.append(float(d[tok].logprob) if tok in d else None)
    glp = []
    for d in o.outputs[0].logprobs:
        items = sorted(d.items(), key=lambda kv: -kv[1].logprob)[:2] if d else []
        glp.append([(int(t), float(x.logprob)) for t, x in items])
    return vals, list(o.outputs[0].token_ids), glp

out = []
for r in rows:
    sid, ids = r['sid'], r['input_ids']
    sched.reset_prefix_cache(); assert sched.block_manager.get_num_free_gpu_blocks() == CAP
    pa, ta, ga = gen_prompt_lp(ids)
    sched.reset_prefix_cache(); assert sched.block_manager.get_num_free_gpu_blocks() == CAP
    pb, tb, gb = gen_prompt_lp(ids)
    first_prompt_diff = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), None)
    max_delta = max((abs(x - y) for x, y in zip(pa, pb) if x is not None and y is not None), default=0.0)
    out.append(dict(sid=sid, block=BLOCK, in_len=len(ids), tokens_equal=ta == tb,
                    first_prompt_logprob_diff=first_prompt_diff,
                    max_prompt_logprob_delta=max_delta,
                    prompt_lp_a=pa, prompt_lp_b=pb, tokens_a=ta, tokens_b=tb))
    print(sid, 'tokens_equal', ta == tb, 'first_prefill_diff', first_prompt_diff, 'max_delta', max_delta, flush=True)

json.dump(dict(variant='F', block=BLOCK, capacity=CAP, samples=out),
          open(f'records/stage1_F_b{BLOCK}.json', 'w'), indent=1)
print('DONE F b%d' % BLOCK, flush=True)
