"""Device-explicit adapter around the frozen scientific reference equations."""
from collections import Counter
import copy
import json
from pathlib import Path
import time
import zlib

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from huggingface_hub import snapshot_download
from safetensors import safe_open

from . import ROOT
from attack_lib import probe_ids, cluster_states, recover_secrets, ground_truth_errors
from corrected.legacy import recover_secrets as legacy_recover
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak
from run_suite import NATURAL


def stable_seed(sid, purpose='target'):
    return zlib.crc32(('corrected-20261029-v1:'+purpose+':'+str(sid)).encode())


def seed_all(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def clone_cache(cache):
    return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k,v in cache))


def first_cache(model, ids, **kwargs):
    """Run native first layer, stopping only after its cache has been written."""
    cache = DynamicCache()
    class StopFirst(Exception):
        pass
    def stop(*_):
        raise StopFirst()
    hook = model.model.layers[0].register_forward_hook(stop)
    try:
        try:
            model(input_ids=ids, past_key_values=cache, use_cache=True, **kwargs)
        except StopFirst:
            pass
    finally:
        hook.remove()
    assert len(cache) == 1
    return cache


def checkpoint(spec, cache_dir, offline=False):
    return Path(snapshot_download(spec['repo'], revision=spec['revision'],
                                  cache_dir=str(cache_dir), local_files_only=offline,
                                  allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','merges.txt','vocab.json','LICENSE','README.md']))


def load_first_layer(snapshot, dtype):
    """Load genuine checkpoint embeddings/layer zero, not synthetic weights."""
    source_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config = copy.deepcopy(source_config)
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype, attn_implementation='eager')
    model.lm_head = torch.nn.Identity()
    model.model.norm = torch.nn.Identity()
    required = set(model.state_dict())
    index_path = snapshot/'model.safetensors.index.json'
    if index_path.exists():
        mapping = json.loads(index_path.read_text())['weight_map']
        groups = {}
        for name in required:
            assert name in mapping, ('checkpoint tensor absent', name)
            groups.setdefault(mapping[name], []).append(name)
    else:
        groups = {'model.safetensors': sorted(required)}
    weights = {}
    for filename, names in groups.items():
        with safe_open(snapshot/filename, framework='pt') as handle:
            for name in names:
                weights[name] = handle.get_tensor(name).to(dtype=dtype)
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval()
    model._source_config = source_config
    return model


@torch.inference_mode()
def native_gate(snapshot, dtype, device, input_arrays, partial=False):
    """Independent native-full-forward versus first-layer cache comparison."""
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
        torch_dtype=dtype, attn_implementation='eager', low_cpu_mem_usage=True).to(device).eval()
    references = []
    for raw in input_arrays:
        ids = torch.tensor([raw], device=device)
        full = model(input_ids=ids, use_cache=True).past_key_values
        reference = tuple(t.clone() for t in full[0])
        short = first_cache(model, ids)
        assert all(torch.equal(x,y) for x,y in zip(reference,short[0])), 'Native early-stop cache mismatch'
        references.append(reference)
    if partial:
        del model
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
        model = load_first_layer(snapshot, dtype).to(device)
        for raw, reference in zip(input_arrays,references):
            short = first_cache(model, torch.tensor([raw],device=device))
            assert all(torch.equal(x,y) for x,y in zip(reference,short[0])), 'Checkpoint first-layer loading mismatch'
    return model, dict(native_full_vs_early_stop_exact=True,
        partial_checkpoint_vs_full_exact=partial, inputs=input_arrays,
        model_dtype=str(dtype), device=str(device), gate_inputs=len(input_arrays))


class Harness:
    def __init__(self, spec, *, device='cpu', dtype='float32', block=16, key_seed=42,
                 calibrator='corrected', cache_dir=None, strategy='full', threads=8,
                 offline=False, external_gate=None, calibration=True):
        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = getattr(torch,dtype)
        self.B = block
        self.calibrator = calibrator
        self.cache_dir = Path(cache_dir or ROOT/'model_cache/hub')
        self.snapshot = checkpoint(spec, self.cache_dir, offline)
        self.tok = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        config = AutoConfig.from_pretrained(self.snapshot, local_files_only=True)
        self.L = config.num_hidden_layers
        self.H = config.num_key_value_heads
        self.D = getattr(config,'head_dim',None) or config.hidden_size//config.num_attention_heads
        gates = [self.tok(NATURAL[i]).input_ids[:16] for i in (0,1,2)]
        if strategy == 'first_layer' and external_gate:
            gate = json.loads(Path(external_gate).read_text())
            assert gate['revision'] == spec['revision'] and gate['partial_checkpoint_vs_full_exact']
            self.model = load_first_layer(self.snapshot,self.dtype).to(self.device)
            self.gate = dict(external_native_cpu_gate=gate, resident_layers=1)
        else:
            self.model,self.gate = native_gate(self.snapshot,self.dtype,self.device,gates,partial=strategy=='first_layer')
        self.strategy = strategy
        self.theta = {}
        for li in range(self.L):
            self.theta[f'layer_{li}']={'key_max_values':[0.]*self.H,'value_max_values':[0.]*self.H}
        maxima = None
        for text in NATURAL:
            cache = self.cache(self.ids(text,complete=False))
            current={name:t[0].abs().amax(dim=(1,2)).float().cpu() for name,t in zip(['key_max_values','value_max_values'],cache[0])}
            maxima=current if maxima is None else {k:torch.maximum(maxima[k],v) for k,v in current.items()}
        self.theta['layer_0']={k:v.tolist() for k,v in maxima.items()}
        self.public_model = self.model
        self._dictionary()
        self.reset_key(key_seed)
        if calibration:
            self.secrets,self.calibration = self.calibrate()

    def ids(self,text,complete=True):
        ids=self.tok(text,return_tensors='pt').input_ids.to(self.device)
        return ids[:,:ids.shape[1]//self.B*self.B] if complete else ids

    def cache(self,ids,model=None,**kwargs):
        return first_cache(model or self.model,ids,**kwargs)

    def _dictionary(self):
        model=self.public_model
        projection=model.model.layers[0].self_attn.v_proj
        embedding=model.get_input_embeddings().weight
        ids=self.tok(NATURAL[0],return_tensors='pt').input_ids.to(self.device)
        captured={}
        hook=projection.register_forward_pre_hook(lambda module,args:captured.update(x=args[0].detach().clone()))
        try:
            actual_cache=self.cache(ids)
        finally:
            hook.remove()
        layer=model.model.layers[0]
        choices=[(name,getattr(layer,name)) for name in ['input_layernorm','layer_norm'] if hasattr(layer,name)]
        choices.append(('identity',None))
        errors=[]
        for name,module in choices:
            plain=embedding[ids[0]]
            value=module(plain) if module is not None else plain
            errors.append((float((value-captured['x'][0]).abs().max()),name,module))
        norm_error,norm_name,norm=min(errors,key=lambda item:item[0])
        assert norm_error==0., ('Public input normalization is not represented',norm_error)
        rows=[]
        for start in range(0,len(embedding),2048):
            # Use the actual module, including its native bias/rounding semantics.
            e=embedding[start:start+2048]
            rows.append(projection(norm(e) if norm is not None else e).float().reshape(-1,self.H,self.D))
        self.dictionary=torch.cat(rows)
        self.normalized=torch.nn.functional.normalize(self.dictionary,dim=-1)
        actual=actual_cache[0][1][0].transpose(0,1).float()
        expected=self.dictionary[ids[0]]
        rel=(actual-expected).norm()/actual.norm().clamp_min(1e-20)
        assert float(rel)<(1e-4 if self.dtype==torch.float32 else .01), ('dictionary/cache mismatch',float(rel))
        self.dictionary_validation={'relative_vector_error':float(rel),'tokens':ids.shape[1],
                                    'input_normalization':norm_name,'normalization_input_error':norm_error}

    def reset_key(self,key_seed,ratio=1.,hold_column=None):
        seed_all(key_seed)
        self.cfg=get_kvcloak_config(self.L,self.H,self.D,self.B,self.theta,ratio,ratio,2.)
        if hold_column is not None:
            for li in range(self.L):
                for h in range(self.H):
                    for kv in (0,1):
                        for key in ['M_angles','M_ratios','M_ratio']:
                            self.cfg[li][h][kv][key]=copy.deepcopy(hold_column[li][h][kv][key])
        self.cloak=KVCloak(self.cfg,dtype=torch.bfloat16,fused=False,need_ratio=ratio!=1.,add_a=True)
        self.key_seed=key_seed

    def protect(self,cache,seed,cloak=None):
        seed_all(seed)
        # Official BF16 operations reuse and overwrite their input storage.
        return (cloak or self.cloak).obfuscate(clone_cache(cache))

    def calibrate(self,cloak=None,known_model=None):
        import math
        target=max(1024,int(2.5*self.B*sum(1/k for k in range(1,self.B+1)))*self.B)
        ids,prefix,char=probe_ids(self.tok,target=target,max_positions=getattr(self.public_model.config,'max_position_embeddings',None))
        ids=ids.to(self.device)
        public=self.cache(ids,model=self.public_model)[0][1][0].float().cpu().clone()
        defended=self.cache(ids,model=known_model or self.model)
        protected=self.protect(defended,1234,cloak)
        start=(prefix+self.B-1)//self.B
        count=ids.shape[1]//self.B
        secrets={};rows=[]
        fn=legacy_recover if self.calibrator=='legacy' else recover_secrets
        for h in range(self.H):
            blocks=protected[0][1][0,h].float().cpu().reshape(-1,self.B,self.D)[start:count]
            n,centers,intra,inter,_=cluster_states(blocks)
            item={'head':h,'states':n,'complete':n==self.B,'intra':intra,'inter':inter if math.isfinite(inter) else None}
            if n==self.B:
                sec=fn(centers,public[h,start*self.B:count*self.B],self.B)
                item['diagnostics']=ground_truth_errors(sec,self.cfg[0][h][1],public[h,start*self.B:count*self.B])
                secrets[h]={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in sec.items()}
            rows.append(item)
        return secrets,dict(actual_probe_tokens=ids.shape[1],prefix_tokens=prefix,repeat_character=char,heads=rows)

    def decode_block(self,C,sec,h,hypothesis_batch=4):
        best_score=None;best_pred=None
        dictionary=self.normalized[:,h].T
        b=self.B
        for start in range(0,b,hypothesis_batch):
            js=range(start,min(start+hypothesis_batch,b))
            hypotheses=torch.stack([C-torch.outer(sec['s'][j],sec['a_hat']) for j in js])
            X=torch.nn.functional.normalize((sec['s'][None]@hypotheses)@sec['M'].T,dim=-1)
            scores=X@dictionary
            maxima,preds=scores.max(-1)
            sums=maxima.sum(-1)
            j=int(sums.argmax())
            if best_score is None or sums[j].item()>best_score:
                best_score=sums[j].item();best_pred=preds[j].tolist()
        return best_pred

    def score(self,ids,protected,secrets=None,heads=None):
        secrets=self.secrets if secrets is None else secrets
        heads=range(self.H) if heads is None else heads
        truth=ids[0].tolist();result=[]
        for h in heads:
            bags=[];counts=[]
            for k in range(len(truth)//self.B):
                pred=self.decode_block(protected[0][1][0,h,k*self.B:(k+1)*self.B].float(),secrets[h],h) if h in secrets else []
                bags.append(pred);counts.append(sum((Counter(pred)&Counter(truth[k*self.B:(k+1)*self.B])).values()))
            result.append(dict(head=h,bags=bags,block_correct=counts,correct=sum(counts),total=len(truth),calibration_complete=h in secrets))
        return dict(correct=sum(r['correct'] for r in result),total=sum(r['total'] for r in result),heads=result)

    def verify_decoder(self):
        from attack_lib import attack_block
        ids=self.ids(NATURAL[2])
        if ids.shape[1]<self.B:
            ids=self.ids(' '.join(NATURAL))[:,:self.B]
        cache=self.cache(ids)
        protected=self.protect(cache,7001)
        checked=[]
        for h in sorted({0,self.H-1}):
            if h not in self.secrets:continue
            C=protected[0][1][0,h,:self.B].float();sec=self.secrets[h]
            reference=attack_block(C,sec['s'],sec['a_hat'],sec['M'],self.normalized[:,h])
            observed=self.decode_block(C,sec,h)
            assert observed==reference, 'Batched decoder disagrees with frozen serial reference on the predeclared gate'
            checked.append(dict(head=h,reference=reference,observed=observed))
        return dict(equivalent=True,checks=checked,input_ids=ids[0].tolist())

    def environment(self):
        return dict(model=self.spec,device=str(self.device),model_dtype=str(self.dtype),cloak_dtype='bfloat16',
                    source_layers=self.L,resident_layers=len(self.model.model.layers),strategy=self.strategy,
                    attention_implementation='eager',calibrator=self.calibrator,block_size=self.B,
                    heads=self.H,head_dimension=self.D,torch=torch.__version__,
                    transformers=__import__('transformers').__version__,tf32=False,
                    gate=self.gate,dictionary_validation=self.dictionary_validation)
