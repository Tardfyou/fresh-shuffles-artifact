"""Protected host stores; plaintext exists only at the trusted restore boundary."""
from dataclasses import dataclass
import copy,uuid,zlib
import torch
from transformers.cache_utils import DynamicCache
from . import ROOT
import exp_veilkv_plus as codec
from corrected.freshness import KeyService
from kvcloak import KVCloak
from .layout import bits_equal


@dataclass
class Entry:
    packet: object
    context: dict
    table: list
    length: int
    dtype: object
    request: str
    public_tokens: int


class ByteStore:
    def __init__(self,directory,identity):
        self.service=KeyService(directory/'nonce-ledger.sqlite');self.identity=identity;self.public={};self.prefix_reuses=0
    def seal(self,plain,table,tokens,request,principal,public_tokens,epoch):
        h=plain[0][0].shape[1];d=plain[0][0].shape[-1];dtype=plain[0][0].dtype
        ctx=dict(**self.identity,tokens=torch.tensor([tokens]),n_prefix_blocks=public_tokens//16,profile='tee',storage_dtype=dtype,principal=principal.encode(),nonce=uuid.uuid4().bytes,epoch_id=epoch,variant={},bsz=1,H=h,D=d,true_len=len(tokens),key_service=self.service)
        packet=codec.protect_cache(plain,ctx);key=(epoch,tuple(tokens[:public_tokens]));n=public_tokens//16
        if n:
            bank=self.public.get(key)
            if bank is None:self.public[key]=packet
            else:
                for coordinate in packet.blocks:
                    for j in range(n):
                        a,b=packet.blocks[coordinate][j],bank.blocks[coordinate][j]
                        assert a['ct']==b['ct'] and a['tag']==b['tag'],'A cached public prefix changed bytes'
                        packet.blocks[coordinate][j]=b
                self.prefix_reuses+=1
        entry=Entry(packet,ctx,list(table),len(tokens),dtype,request,public_tokens)
        recovered=self.open(entry)
        return entry,bits_equal(plain,recovered)
    def open(self,entry):
        raw=codec.recover_cache(entry.packet,entry.context)
        return tuple(tuple(x.to(entry.dtype) for x in pair) for pair in raw)
    def tamper(self,entry,kind,previous=None):
        packet=copy.deepcopy(entry.packet);ctx=dict(entry.context)
        first_private=ctx['n_prefix_blocks'];first=packet.blocks[(0,1)][first_private]
        if kind in ['ciphertext','tag']:
            key='ct' if kind=='ciphertext' else 'tag';value=first[key];first[key]=bytes([value[0]^1])+value[1:]
        elif kind in ['principal','nonce','model']:
            field={'principal':'principal','nonce':'nonce','model':'model_dig'}[kind];value=ctx[field];ctx[field]=bytes([value[0]^1])+value[1:]
        elif kind=='epoch':packet.epoch+=1
        elif kind=='prefix':
            assert ctx['n_prefix_blocks']>0;first=packet.blocks[(0,1)][0];first['artifact_id']=bytes([first['artifact_id'][0]^1])+first['artifact_id'][1:]
        elif kind=='replay':
            assert previous is not None;packet=previous.packet
        else:raise ValueError(kind)
        entry.packet=packet;entry.context=ctx
    def private_payload(self,entry):return entry.packet.blocks[(0,1)][entry.context['n_prefix_blocks']]['ct']
    def public_payload(self,entry):return entry.packet.blocks[(0,1)][0]['ct']


@dataclass
class Cloaked:
    tensors: tuple
    permutations: list
    public_tokens: int


class CloakStore:
    """Official obfuscation plus legal physical-row restoration using held P.

    The attacker never receives this store's matrices/permutations. Floating
    restore deviations remain measured outcomes; they are not byte-codec passes.
    """
    def __init__(self,configuration,device):
        self.cloak=KVCloak(configuration,torch.float32,False,False,True);self.device=device;self.public={}
        torch.manual_seed(1234)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(1234)
        self.cloak._prepare_device_tensors(torch.device(device))
        self.inverse=[]
        for layer in self.cloak._device_cache[torch.device(device)]['obf']:
            roles=[]
            for role in ['k','v']:
                s=layer['S_'+role+'_batch'][0,:,0].double().cpu();m=layer['M_'+role+'_batch'][0,:,0].double().cpu();a=layer['A_'+role+'_batch'][0,:,0].double().cpu()
                roles.append((torch.linalg.inv(s),torch.linalg.inv(m),a))
            self.inverse.append(roles)
        self.counter=0
    def seal(self,plain,table,tokens,request,principal,public_tokens,epoch):
        from unittest.mock import patch
        self.counter+=1;seed=zlib.crc32(('vllmcompat20261110|'+request+'|'+str(len(tokens))).encode())
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        permutations=[];argsort=torch.argsort
        def record(*args,**kwargs):
            value=argsort(*args,**kwargs);permutations.append(value.detach().cpu().clone());return value
        # Float32 input keeps the official output float32, even with BF16 engine KV.
        supplied=DynamicCache.from_legacy_cache(tuple(tuple(x.to(self.device,torch.float32).clone() for x in pair) for pair in plain))
        with patch.object(torch,'argsort',record):protected=self.cloak.obfuscate(supplied)
        tensors=tuple(tuple(x.cpu().clone() for x in pair) for pair in protected)
        packet=Cloaked(tensors,permutations,public_tokens);n=public_tokens//16;key=tuple(tokens[:public_tokens])
        if n:
            bank=self.public.get(key)
            if bank is None:self.public[key]=packet
            else:
                for li in range(len(tensors)):
                    packet.permutations[li][:n]=bank.permutations[li][:n]
                    for role in [0,1]:packet.tensors[li][role][:,:,:public_tokens].copy_(bank.tensors[li][role][:,:,:public_tokens])
        entry=Entry(packet,dict(shuffle_seed=seed),list(table),len(tokens),plain[0][0].dtype,request,public_tokens)
        restored=self.open(entry);return entry,bits_equal(plain,restored)
    def open(self,entry):
        output=[]
        for li,pair in enumerate(entry.packet.tensors):
            inverse_permutation=torch.argsort(entry.packet.permutations[li],dim=-1)
            roles=[]
            for role,tensor in enumerate(pair):
                heads=tensor.shape[1];dim=tensor.shape[-1];blocks=tensor.shape[2]//16
                c=tensor[0].double().reshape(heads,blocks,16,dim);sinv,minv,a=self.inverse[li][role]
                y=sinv[:,None]@c
                y=torch.gather(y,2,inverse_permutation[None,:,:,None].expand(heads,blocks,16,dim))
                value=((y-a[:,None])@minv[:,None]).reshape(1,heads,-1,dim)[:,:,:entry.length]
                roles.append(value.to(entry.dtype))
            output.append(tuple(roles))
        return tuple(output)
    def private_payload(self,entry):return None
    def public_payload(self,entry):return None
