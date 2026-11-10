"""Logical K/V views of real FlashAttention physical cache pages."""
import torch


def check_layout(caches,block):
    assert caches and all(x.ndim==5 and x.shape[0]==2 and x.shape[2]==block for x in caches),'Expected FlashAttention [2,pages,block,heads,dim]'
    shape=caches[0].shape
    assert all(x.shape==shape and x.dtype==caches[0].dtype for x in caches)
    return dict(layers=len(caches),pages=shape[1],physical_block=block,heads=shape[3],head_dim=shape[4],dtype=str(caches[0].dtype),device=str(caches[0].device),strides=[list(x.stride()) for x in caches])


def gather(caches,table,length,block):
    assert 0<length<=len(table)*block and len(set(table))==len(table)
    ix=torch.tensor(table[:(length+block-1)//block],device=caches[0].device,dtype=torch.long)
    result=[]
    for layer in caches:
        pair=[]
        for role in [0,1]:
            selected=layer[role].index_select(0,ix).reshape(-1,layer.shape[3],layer.shape[4])[:length]
            pair.append(selected.permute(1,0,2).unsqueeze(0).contiguous())
        result.append(tuple(pair))
    return tuple(result)


def write_page(caches,page,plain,logical_page,length,block):
    start=logical_page*block;n=min(block,length-start);assert n>0
    for li,pair in enumerate(plain):
        for role,value in enumerate(pair):caches[li][role,page,:n].copy_(value[0,:,start:start+n].permute(1,0,2).to(caches[li].device,caches[li].dtype))


def zero_pages(caches,pages):
    if not pages:return
    ix=torch.tensor(sorted(set(pages)),device=caches[0].device,dtype=torch.long)
    for layer in caches:layer.index_fill_(1,ix,0)
    if caches[0].is_cuda:torch.cuda.synchronize(caches[0].device)
    assert all(not bool(layer.index_select(1,ix).count_nonzero()) for layer in caches),'GPU plaintext was not evicted'


def bits_equal(a,b):
    return len(a)==len(b) and all(x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.contiguous().view(torch.uint8).cpu(),y.contiguous().view(torch.uint8).cpu()) for p,q in zip(a,b) for x,y in zip(p,q))
