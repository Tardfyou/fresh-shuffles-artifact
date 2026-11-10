"""Cache-policy extension: share public pages, isolate private request pages."""
import uuid
REGISTRY={}
DOMAIN=uuid.uuid4().hex
DEPTH={}
COUNTS={'public_hashes':0,'private_hashes':0}
_INSTALLED=False


def register(request,principal,public_tokens,block):
    assert public_tokens%block==0
    REGISTRY[request]=dict(principal=principal,public_blocks=public_tokens//block,public_tokens=public_tokens)


def scope_extra(base,scope,depth):
    _,domain,public,principal,request,original=scope
    if depth<public:return ('veilkv-public',domain,original)
    return ('veilkv-private',domain,principal,request,original)


def install():
    global _INSTALLED
    if _INSTALLED:return
    from vllm.sequence import Sequence
    from vllm.core.scheduler import Scheduler
    from vllm.core.block.prefix_caching_block import PrefixCachingBlock
    old_add=Scheduler.add_seq_group
    old_extra=Sequence.extra_hash
    old_hash=PrefixCachingBlock.hash_block_tokens.__func__
    def add(scheduler,group):
        conf=REGISTRY[group.request_id]
        for seq in group.get_seqs():seq._veilkv_scope=('veilkv-scope',DOMAIN,conf['public_blocks'],conf['principal'],group.request_id,None)
        return old_add(scheduler,group)
    def extra(seq):
        base=old_extra(seq);tag=getattr(seq,'_veilkv_scope',None)
        return (*tag[:-1],base) if tag is not None else base
    def hashed(cls,is_first_block,prev_block_hash,cur_block_token_ids,extra_hash=None):
        if isinstance(extra_hash,tuple) and extra_hash and extra_hash[0]=='veilkv-scope':
            depth=0 if is_first_block else DEPTH[prev_block_hash]+1
            effective=scope_extra(None,extra_hash,depth)
            result=old_hash(cls,is_first_block,prev_block_hash,cur_block_token_ids,effective)
            DEPTH[result]=depth;COUNTS['public_hashes' if depth<extra_hash[2] else 'private_hashes']+=1
            return result
        return old_hash(cls,is_first_block,prev_block_hash,cur_block_token_ids,extra_hash)
    Scheduler.add_seq_group=add;Sequence.extra_hash=extra;PrefixCachingBlock.hash_block_tokens=classmethod(hashed)
    _INSTALLED=True
