"""Small codec/runtime unit checks, distinct from real-model cohort evidence."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile


def run(runtime,report):
    sys.path.insert(0,str(runtime))
    import torch
    import exp_veilkv_plus as codec
    from corrected.freshness import KeyService
    from veilkv_runtime import recover_for_model
    torch.set_num_threads(1);torch.manual_seed(20261102)
    results=[]
    def clone(cache):return tuple((k.clone(),v.clone()) for k,v in cache)
    def equal(a,b):
        return len(a)==len(b) and all(x.shape==y.shape and x.dtype==y.dtype and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for ap,bp in zip(a,b) for x,y in zip(ap,bp))
    with tempfile.TemporaryDirectory(prefix='codec-unit-') as directory:
        service=KeyService(Path(directory)/'ledger.sqlite')
        for dtype in (torch.float32,torch.bfloat16):
            for batch in (1,2):
                for length in (48,51):
                    for public in (0,1):
                        index=len(results);cache=tuple((torch.randn(batch,2,length,8).to(dtype),torch.randn(batch,2,length,8).to(dtype)) for _ in range(2))
                        cache[0][0].flatten()[0]=-0.0;cache[0][1].flatten()[0]=torch.finfo(dtype).tiny
                        before=clone(cache)
                        ctx={'model_dig':b'unit-model-'+str(index).encode(),'tok_dig':b'unit-tokenizer','rope':b'unit-rope','tokens':torch.arange(batch*length).reshape(batch,length),'n_prefix_blocks':public,'profile':'tee','storage_dtype':dtype,'principal':b'unit-principal','nonce':index.to_bytes(16,'big'),'epoch_id':3,'variant':{},'bsz':batch,'H':2,'D':8,'true_len':length,'key_service':service}
                        protected=codec.protect_cache(cache,ctx)
                        recovered=tuple(recover_for_model(protected,ctx,model_dtype=dtype))
                        exact=equal(before,recovered);unmodified=equal(before,cache)
                        changed={**ctx,'principal':b'another-principal'}
                        try:codec.recover_cache(protected,changed);reject=False
                        except codec.TamperError:reject=True
                        results.append({'dtype':str(dtype),'batch':batch,'length':length,'public_blocks':public,'cache_bits_exact':exact,'source_unmodified':unmodified,'principal_only_rejected':reject,'pass':exact and unmodified and reject})
    result={'unit_cases':len(results),'passed':sum(r['pass'] for r in results),'all_pass':all(r['pass'] for r in results),'scope':'Synthetic finite-tensor codec/runtime unit checks only; not additional WildChat prompts or serving/GPU evidence.','dependencies':{name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','cryptography']},'cases':results}
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if not result['all_pass']:raise SystemExit('Codec/runtime unit check failed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();run(args.runtime,args.report)
