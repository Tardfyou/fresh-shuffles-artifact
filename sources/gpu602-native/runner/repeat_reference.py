"""Existing A/C repeat-reference protocol; only the frozen prompt cohort grows."""
import json
import os
import time

import torch

from .core import checkpoint, hardware_record
from .worker import samples_for, write


def run(job, args, store):
    os.environ['VLLM_USE_V1'] = '0'
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    path = checkpoint(job['model'], args.cache_dir, args.offline)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    rows = samples_for(job)
    inputs = [(row, tok(row['text']).input_ids) for row in rows]
    assert all(len(ids) + 16 <= 1024 for _, ids in inputs)
    block = job['block']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision('highest')
    engine = LLM(
        model=str(path), tokenizer=str(path), dtype='bfloat16',
        tensor_parallel_size=1, enforce_eager=True,
        enable_prefix_caching=job['variant'] == 'C', block_size=block,
        max_model_len=1024, max_num_seqs=1,
        num_gpu_blocks_override={16: 72, 32: 40}[block],
        gpu_memory_utilization=.5, distributed_executor_backend='uni')
    scheduler = engine.llm_engine.scheduler[0]
    capacity = scheduler.block_manager.get_num_free_gpu_blocks()
    params = SamplingParams(temperature=0., max_tokens=16)
    write(store.root/'environment.json', dict(
        engine='vLLM', version=__import__('vllm').__version__,
        device=str(args.device), model_dtype='torch.bfloat16',
        model=job['model'], hardware=hardware_record(args.device),
        torch=torch.__version__, transformers=__import__('transformers').__version__,
        variant=job['variant'], round=job['round'], block_size=block,
        max_model_len=1024, allocated_cache_blocks={16: 72, 32: 40}[block],
        protocol='Original diag_stage2 A/C, three independent engine rounds',
        prefix_cache_enabled=job['variant'] == 'C',
        reset_before_pair=job['variant'] == 'C', temperature=0., max_tokens=16))
    for row, ids in inputs:
        if row['sid'] in store.done:
            continue
        reset_verified = None
        if job['variant'] == 'C':
            ok = bool(scheduler.reset_prefix_cache())
            free = scheduler.block_manager.get_num_free_gpu_blocks()
            reset_verified = ok and free == capacity
            assert reset_verified, 'Prefix-cache reset failed'
        start = time.monotonic()
        a = list(engine.generate([{'prompt_token_ids': ids}], params,
                                use_tqdm=False)[0].outputs[0].token_ids)
        b = list(engine.generate([{'prompt_token_ids': ids}], params,
                                use_tqdm=False)[0].outputs[0].token_ids)
        first = None if a == b else next(
            (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
            min(len(a), len(b)))
        store.append(row['sid'], status='measured', input_ids=ids,
                     variant=job['variant'], round=job['round'],
                     a=a, b=b, equal=a == b, first_div=first,
                     reset_verified=reset_verified, seconds=time.monotonic()-start)
        print(job['id'], row['sid'], a == b, flush=True)
    return store.finish([row['sid'] for row in rows])
