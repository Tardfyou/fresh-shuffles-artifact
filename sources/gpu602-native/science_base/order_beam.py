import torch
from collections import Counter

def beam_cached(ms,ctx,m,width=16):
 """Exact autoregressive beam search on a fixed prefix, with reordered KV caches.

 Caller retains <=80 preceding tokens, so a 16-token block stays within the
 96-token input limit. Unlike rolling truncation this admits exact KV reuse.
 """
 if not ms:return []
 assert len(ctx)<=80 and sum(ms.values())==16
 beams=[(0.,tuple(ctx),Counter(ms))];start=len(ctx)
 with torch.no_grad():
  output=m.model(input_ids=torch.tensor([ctx],device=next(m.parameters()).device),use_cache=True)
  cache=output.past_key_values
  lp=torch.log_softmax(m.lm_head(output.last_hidden_state[:,-1]).float(),dim=-1)
  for step in range(sum(ms.values())):
   candidates={}
   for i,(sc,sq,mx) in enumerate(beams):
    for t,n in mx.items():
     if n<=0:continue
     rem=mx.copy();rem[t]-=1
     if not rem[t]:del rem[t]
     ns=sq+(t,);score=sc+lp[i,t].item();key=(ns,tuple(sorted(rem.items())))
     if key not in candidates or score>candidates[key][0]:candidates[key]=(score,ns,rem,i,t)
   chosen=sorted(candidates.values(),key=lambda b:(-b[0],b[1]))[:width]
   beams=[entry[:3] for entry in chosen]
   if step==sum(ms.values())-1:break
   parents=torch.tensor([entry[3] for entry in chosen],device=next(m.parameters()).device)
   cache.reorder_cache(parents)
   tokens=torch.tensor([[entry[4]] for entry in chosen],device=next(m.parameters()).device)
   output=m.model(input_ids=tokens,past_key_values=cache,use_cache=True)
   cache=output.past_key_values
   lp=torch.log_softmax(m.lm_head(output.last_hidden_state[:,-1]).float(),dim=-1)
 result=list(beams[0][1][start:]);assert Counter(result)==Counter(ms)
 return result
