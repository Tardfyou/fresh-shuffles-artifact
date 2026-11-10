"""Single-GPU live engine control; no model inference runs on the CPU."""
import os,time,json,uuid,base64
from pathlib import Path
from types import SimpleNamespace
import torch
from . import ROOT
from . import scope_policy


def synchronized():torch.cuda.synchronize()


def setup(spec,block,settings,folder):
    os.environ['VLLM_USE_V1']='0';os.environ['VLLM_ATTENTION_BACKEND']='FLASH_ATTN';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['NVIDIA_TF32_OVERRIDE']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    assert torch.cuda.is_available(),'This experiment requires CUDA; do not substitute CPU model inference'
    import vllm,transformers
    assert vllm.__version__=='0.8.5.post1' and torch.__version__.split('+')[0]=='2.6.0' and transformers.__version__=='4.51.3'
    torch.set_num_threads(settings['cpu_threads']);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.set_float32_matmul_precision('highest')
    from .api_guard import check
    upstream_api=check()
    scope_policy.install()
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig,AutoTokenizer
    cache=Path(settings['cache_dir']).expanduser();cache=cache if cache.is_absolute() else ROOT/cache
    path=snapshot_download(spec['repo'],revision=spec['revision'],cache_dir=str(cache),allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','merges.txt','vocab.json','LICENSE','README.md'])
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True);config=AutoConfig.from_pretrained(path,local_files_only=True)
    from .identity import checkpoint_identity
    identity=checkpoint_identity(SimpleNamespace(config=config),tok,spec)
    from vllm import LLM
    model=LLM(model=path,tokenizer=path,dtype='bfloat16',tensor_parallel_size=1,distributed_executor_backend='uni',worker_cls='integration.worker.VeilKVWorker',enforce_eager=True,enable_prefix_caching=True,block_size=block,max_model_len=4096,max_num_seqs=2,max_num_batched_tokens=4096,num_gpu_blocks_override=512,gpu_memory_utilization=settings['gpu_memory_utilization'],swap_space=0,enable_chunked_prefill=False,num_scheduler_steps=1,disable_async_output_proc=True,seed=42)
    worker=model.llm_engine.model_executor.driver_worker.worker
    from .worker import VeilKVWorker
    assert isinstance(worker,VeilKVWorker)
    layout=worker.configure_integration(str(folder/'state'),identity)
    env=dict(spec=spec,torch=torch.__version__,cuda_runtime=torch.version.cuda,transformers=transformers.__version__,vllm=vllm.__version__,gpu=torch.cuda.get_device_name(),total_gpu_memory=torch.cuda.get_device_properties(0).total_memory,model_dtype='bfloat16',protected_byte_dtype='bfloat16',kvcloak_operation_and_storage_dtype='float32',physical_layout=layout,engine='V0 UniProc FlashAttention eager',prefix_scope_policy_installed=True,identity_algorithm='pinned repo/revision/tokenizer/RoPE labels',gpu_memory_utilization=settings['gpu_memory_utilization'],upstream_api=upstream_api,scope_domain=scope_policy.DOMAIN,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return model,worker,tok,env


def clear(model,worker,mode):
    assert not model.llm_engine.has_unfinished_requests()
    assert model.llm_engine.reset_prefix_cache(),'Prefix cache still has referenced pages'
    worker.bridge.reset();worker.bridge.configure(mode);worker.logit_frames=[];worker.bridge.capture_requests=set();worker.bridge.full_capture_requests=set()


def submit(model,worker,requests,*,public_tokens,block,new_tokens=6,capture=False,full_capture=False):
    from vllm import SamplingParams
    for request,principal,tokens in requests:
        scope_policy.register(request,principal,public_tokens,block)
        if capture:worker.bridge.capture_requests.add(request)
        if full_capture:worker.bridge.full_capture_requests.add(request)
        model.llm_engine.add_request(request,{'prompt_token_ids':tokens},SamplingParams(temperature=0.,max_tokens=new_tokens,ignore_eos=True))


def drain(model,worker,requests,epoch_rotate=False):
    outputs={};frames=[];step=0
    while model.llm_engine.has_unfinished_requests():
        before=len(worker.logit_frames);items=model.llm_engine.step();frames+=worker.logit_frames[before:];step+=1
        for item in items:
            if item.finished:
                assert len(item.outputs)==1
                outputs[item.request_id]=dict(tokens=list(item.outputs[0].token_ids),prompt_token_ids=list(item.prompt_token_ids),cached_tokens=item.num_cached_tokens,finish_reason=item.outputs[0].finish_reason)
        if epoch_rotate and step==2:worker.bridge.epoch=4
    assert set(outputs)=={r[0] for r in requests}
    return outputs,frames


def run(model,worker,requests,*,public_tokens,block,mode,prefix=None,epoch_rotate=False):
    clear(model,worker,mode)
    if prefix:
        warm=[(requests[0][0]+'-warm','public-prefix',prefix)];submit(model,worker,warm,public_tokens=public_tokens,block=block,new_tokens=1);drain(model,worker,warm);worker.logit_frames=[]
    submit(model,worker,requests,public_tokens=public_tokens,block=block)
    out,frames=drain(model,worker,requests,epoch_rotate=epoch_rotate)
    payload={r[0]:worker.bridge.payloads(r[0]) for r in requests} if mode=='veilkv' else {}
    return dict(outputs=out,frames=frames,bridge=worker.bridge.report(),payloads=payload)


def compare(a,b,requests):
    assert len(a['frames'])==len(b['frames'])
    results=[]
    for first,second in zip(a['frames'],b['frames']):
        assert set(first)==set(second)
        for request in first:
            x,y=first[request],second[request];d=(x-y).abs();worst=int(d.argmax());results.append(dict(request=request,maximum_absolute_error=float(d[worst]),worst_plain=float(x[worst]),worst_observed=float(y[worst]),worst_index=worst,exact=torch.equal(x,y),plain_argmax=int(x.argmax()),observed_argmax=int(y.argmax())))
    tokens_equal=all(a['outputs'][r[0]]['tokens']==b['outputs'][r[0]]['tokens'] for r in requests)
    return dict(tokens_equal=tokens_equal,logits_exact=all(r['exact'] for r in results),logit_comparisons=results)


def compact(run):return {k:v for k,v in run.items() if k!='frames'}
