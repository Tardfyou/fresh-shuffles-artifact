"""Seal real physical pages after a vLLM call and restore before consumption."""
from pathlib import Path
import base64,json,uuid
import torch
from .layout import check_layout,gather,write_page,zero_pages,bits_equal
from .stores import ByteStore,CloakStore
from . import scope_policy


class Bridge:
    def __init__(self,caches,block,directory,identity):
        self.caches=caches;self.block=block;self.layout=check_layout(caches,block);self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.identity=identity;self.mode='plain';self.byte_store=ByteStore(self.directory,identity);self.cloak_store=None;self.refs={};self.objects={};self.previous={};self.epoch=3;self.steps=[];self.observations={};self.capture_requests=set();self.full_capture={};self.full_capture_requests=set();self.last_entries={};self.pending_fault=None;self.restore_calls=0;self.rejections=0;self.model_calls=0
    def configure(self,mode):
        assert mode in ['plain','veilkv','kvcloak'];assert not self.refs,'Restore/reset old state before switching modes'
        self.mode=mode;self.epoch=3;self.steps=[];self.observations={};self.previous={};self.last_entries={};self.full_capture={};self.pending_fault=None;self.byte_store.public={};self.byte_store.prefix_reuses=0;self.restore_calls=0;self.restored_objects=0;self.rejections=0;self.model_calls=0
    def reset(self):
        # The engine has no active requests at this boundary. Clear free pages
        # too, so an earlier plain control does not leave an unsealed shadow.
        zero_pages(self.caches,list(range(self.layout['pages'])));self.refs={};self.objects={};self.last_entries={};self.previous={};self.byte_store.public={}
        if self.cloak_store is not None:self.cloak_store.public={}
    def configure_cloak(self,cfg):self.cloak_store=CloakStore(cfg,self.caches[0].device)
    @property
    def store(self):return self.byte_store if self.mode=='veilkv' else self.cloak_store
    def before(self):
        if self.mode=='plain':return
        try:
            # Authenticate all objects before writing any plaintext back to GPU.
            restored={key:self.store.open(entry) for key,entry in self.objects.items()}
            for page,(key,j) in self.refs.items():
                entry=self.objects[key];plain=restored[key];write_page(self.caches,page,plain,j,entry.length,self.block)
                n=min(self.block,entry.length-j*self.block)
                actual=gather(self.caches,[page],n,self.block)
                expected=tuple(tuple(t[:,:,j*self.block:j*self.block+n] for t in pair) for pair in plain)
                assert bits_equal(actual,expected),'Restored physical GPU bytes differ from authenticated payload'
            self.restore_calls+=1
            self.restored_objects+=len(restored)
        except Exception:
            zero_pages(self.caches,list(self.refs));self.rejections+=1;raise
    def after(self,metadata):
        snapshots=[];current={};step=dict(mode=self.mode,epoch=self.epoch,sequences=[],roundtrip_bytes_exact=True,zeroized_at_rest=False)
        for group in metadata:
            conf=scope_policy.REGISTRY[group.request_id]
            assert hasattr(group,'seq_data'),'Full V0 metadata is required'
            for sid,data in group.seq_data.items():
                tokens=list(data.get_token_ids());table=list(group.block_tables[sid]);public=min(conf['public_tokens'],len(tokens));public=public//self.block*self.block
                plain=gather(self.caches,table,len(tokens),self.block)
                snapshots.append((group.request_id,sid,tokens,table,public,conf['principal'],plain))
                step['sequences'].append(dict(request=group.request_id,sequence=sid,tokens=len(tokens),input_ids=tokens,physical_blocks=table,public_tokens=public,computed_blocks=list(group.computed_block_nums or [])))
                for j,page in enumerate(table[:(len(tokens)+self.block-1)//self.block]):
                    if page in current:
                        other=current[page]
                        assert j*self.block<public and other['public'],'Private physical page shared across requests'
                    current[page]=dict(public=j*self.block<public)
        if self.mode=='plain':
            for request,sid,tokens,table,public,principal,plain in snapshots:
                if request in self.full_capture_requests:self.full_capture[request]=tuple(tuple(x.cpu() for x in p) for p in plain)
            self.steps.append(step);return
        try:
            for request,sid,tokens,table,public,principal,plain in snapshots:
                host=tuple(tuple(x.cpu() for x in p) for p in plain)
                entry,exact=self.store.seal(host,table,tokens,request,principal,public,self.epoch)
                key=uuid.uuid4().hex;self.objects[key]=entry;step['roundtrip_bytes_exact'] &= exact
                if request in self.last_entries:self.previous[request]=self.last_entries[request]
                self.last_entries[request]=entry
                for j,page in enumerate(table[:(len(tokens)+self.block-1)//self.block]):self.refs[page]=(key,j)
                if request in self.capture_requests and request not in self.observations:
                    assert self.mode=='kvcloak';self.observations[request]=dict(length=len(tokens),protected_value=entry.packet.tensors[0][1][0].clone())
            used={key for key,j in self.refs.values()};self.objects={key:entry for key,entry in self.objects.items() if key in used}
        finally:
            zero_pages(self.caches,list(set(self.refs)|set(current)));step['zeroized_at_rest']=True;self.steps.append(step)
    def inject(self,kind,request):
        assert self.mode=='veilkv';entry=self.last_entries[request];self.byte_store.tamper(entry,kind,self.previous.get(request))
    def payloads(self,request):
        entry=self.last_entries[request]
        if self.mode!='veilkv':return {}
        private=self.byte_store.private_payload(entry) if entry.length>entry.public_tokens else None
        public=self.byte_store.public_payload(entry) if entry.public_tokens else None
        return dict(private=base64.b64encode(private).decode() if private else None,public=base64.b64encode(public).decode() if public else None)
    def report(self):
        return dict(mode=self.mode,layout=self.layout,steps=self.steps,restore_calls=self.restore_calls,restored_objects=self.restored_objects,auth_rejections=self.rejections,model_calls=self.model_calls,stored_prefix_reuses=self.byte_store.prefix_reuses,private_scope_hashes=scope_policy.COUNTS.copy(),protected_host_objects=len(self.objects),physical_pages_sealed=len(self.refs))
