"""Bounded large-model byte-cache check on the frozen selected cohort."""
import hashlib
import uuid
import torch
from .core import synchronize
from .full_model import Consumer,clone,same_cache,logits_diff,SEGMENTS
from .worker import samples_for,write
from corrected.freshness import KeyService
import exp_veilkv_plus as codec


def run(job,args,store):
    c=Consumer(job,args);rows=samples_for(job);service=KeyService(store.root/'freshness.sqlite');invocation=uuid.uuid4().hex
    write(store.root/'environment.json',dict(c.metadata(),public_prefix_tokens=0,input_construction='Frozen selected WildChat texts, first140 model-tokenizer tokens, last6 held out for2/3/1 continuation'))
    for row in rows:
        if row['sid'] in store.done:continue
        tokens=c.tokenizer(row['text']).input_ids[:140];assert len(tokens)>6
        fix=dict(ids=tokens[:-6],cont=tokens[-6:],mask=[1]*(len(tokens)-6),pos=list(range(len(tokens)-6)))
        base=c.baseline(fix);assert all(base['sanity']),'Independent plain controls differ'
        source=clone(base['states'][0]);before=clone(source)
        ctx=dict(**c.dig,tokens=torch.tensor([fix['ids']]),n_prefix_blocks=0,profile='tee',storage_dtype=c.dtype,
                 principal=b'large-validation-'+row['sid'].encode(),nonce=hashlib.blake2b((invocation+row['sid']).encode(),digest_size=16).digest(),
                 epoch_id=3,variant={},bsz=1,H=c.H,D=c.D,true_len=len(fix['ids']),key_service=service)
        obj=codec.protect_cache(tuple((a.cpu(),b.cpu()) for a,b in source),ctx)
        raw=codec.recover_cache(obj,ctx);state=tuple((a.to(c.device,c.dtype),b.to(c.device,c.dtype)) for a,b in raw)
        checks=dict(roundtrip_bytes=same_cache(source,state),source_unmodified=same_cache(source,before),independent_plain=True)
        for field in ['model_dig','tok_dig','rope','principal']:
            altered={**ctx,field:ctx[field]+b'-changed'}
            try:codec.recover_cache(obj,altered);checks[field+'_only_rejected']=False
            except codec.TamperError:checks[field+'_only_rejected']=True
        segments=[]
        for k,(off,n) in enumerate(SEGMENTS):
            state,lg=c.consume(state,fix,off,n);difference=logits_diff(lg,base['logits'][k]);checks[f'segment{k}_logits']=difference['exact'];checks[f'segment{k}_cache']=same_cache(state,base['states'][k+1])
            segments.append(dict(segment=k+1,logits=difference,full_vs_cached=base['full_vs_cached'][k]))
        synchronize(c.device)
        store.append(row['sid'],status='measured',input_ids=fix['ids'],continuation=fix['cont'],public_prefix_tokens=0,checks=checks,segments=segments,functional_pass=all(checks.values()))
    return store.finish([r['sid'] for r in rows])
