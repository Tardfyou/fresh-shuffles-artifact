"""Device-explicit adapter around the frozen scientific reference equations."""
from collections import Counter
import copy
import json
from pathlib import Path
import time
import zlib
import os
import subprocess

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


def load_full(snapshot,dtype,device):
    device=torch.device(device)
    kwargs=dict(local_files_only=True,torch_dtype=dtype,attn_implementation='eager',low_cpu_mem_usage=True)
    if device.type=='cuda':kwargs['device_map']={'':str(device)}
    model=AutoModelForCausalLM.from_pretrained(snapshot,**kwargs)
    if device.type!='cuda':model=model.to(device)
    return model.eval()


def stable_seed(sid, purpose='target'):
    return zlib.crc32(('corrected-20261029-v1:'+purpose+':'+str(sid)).encode())


def seed_all(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def hardware_record(device):
    device=torch.device(device)
    if device.type=='cuda':
        prop=torch.cuda.get_device_properties(device)
        result=dict(type='cuda',name=prop.name,total_memory_bytes=prop.total_memory,
                    compute_capability=[prop.major,prop.minor],visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        try:
            index=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
            result['driver_version']=subprocess.check_output(['nvidia-smi','--id='+index,'--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
        except Exception:result['driver_version']=None
        return result
    model=None
    path=Path('/proc/cpuinfo')
    if path.exists():
        model=next((line.split(':',1)[1].strip() for line in path.read_text().splitlines() if line.startswith('model name')),None)
    return dict(type='cpu',model=model,threads=torch.get_num_threads(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)


def clone_cache(cache):
    return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k,v in cache))


def first_cache(model, ids, **kwargs):
    """Stop immediately after the native cache update, before unused attention.

    Native-full-forward gates verify the produced K/V tensors. Avoiding the
    unused quadratic attention matrix keeps long probes within an 80-GiB card.
    """
    cache = DynamicCache()
    class StopFirst(Exception):
        pass
    original_update=cache.update
    def update(key_states,value_states,layer_idx,cache_kwargs=None):
        result=original_update(key_states,value_states,layer_idx,cache_kwargs)
        if layer_idx==0:raise StopFirst()
        return result
    cache.update=update
    try:
        try:
            model(input_ids=ids, past_key_values=cache, use_cache=True, **kwargs)
        except StopFirst:
            pass
    finally:
        cache.update=original_update
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
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
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
    model = load_full(snapshot,dtype,device)
    references = []
    for raw in input_arrays:
        ids = torch.tensor([raw], device=device)
        full = model(input_ids=ids, use_cache=True).past_key_values
        reference = tuple(t.clone() for t in full[0])
        short = first_cache(model, ids)
        assert all(torch.equal(x,y) for x,y in zip(reference,short[0])), 'Native early-stop cache mismatch'
        backbone=model.model(input_ids=ids,use_cache=True).past_key_values
        assert all(torch.equal(x,y) for x,y in zip(reference,backbone[0])), 'Native backbone cache mismatch'
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
        native_backbone_cache_exact=True,model_dtype=str(dtype), device=str(device), gate_inputs=len(input_arrays))


class Harness:
    def __init__(self, spec, *, device='cpu', dtype='float32', block=16, key_seed=42,
                 calibrator='corrected', cache_dir=None, strategy='full', threads=8,
                 offline=False, external_gate=None, calibration=True, configuration_layers=None,
                 cloak_dtype='bfloat16',cache_forward='first_layer'):
        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision('highest')
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = getattr(torch,dtype)
        self.cloak_dtype=getattr(torch,cloak_dtype)
        self.B = block
        self.calibrator = calibrator
        self.cache_forward=cache_forward
        if strategy=='first_layer':assert cache_forward=='first_layer'
        self.cache_dir = Path(cache_dir or ROOT/'model_cache/hub')
        self.snapshot = checkpoint(spec, self.cache_dir, offline)
        self.tok = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        config = AutoConfig.from_pretrained(self.snapshot, local_files_only=True)
        self.L = config.num_hidden_layers
        self.config_layers = self.L if configuration_layers is None else configuration_layers
        self.H = config.num_key_value_heads
        self.D = getattr(config,'head_dim',None) or config.hidden_size//config.num_attention_heads
        gates = [self.tok(NATURAL[i]).input_ids[:16] for i in (0,1,2)]
        if strategy == 'first_layer' and external_gate:
            gate_path=Path(external_gate)
            if not gate_path.is_absolute():gate_path=ROOT/gate_path
            gate = json.loads(gate_path.read_text())
            assert gate['revision'] == spec['revision'] and gate['partial_checkpoint_vs_full_exact']
            assert gate['model_dtype']=='torch.'+dtype
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
        selected=model or self.model
        if self.cache_forward=='full_model':
            # Execute every native transformer layer, omitting only the unused
            # vocabulary head. The native gate verifies identical first K/V.
            cache=selected.model(input_ids=ids,use_cache=True,**kwargs).past_key_values
            return DynamicCache.from_legacy_cache(((cache[0][0],cache[0][1]),))
        return first_cache(selected,ids,**kwargs)

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
        self.cfg=get_kvcloak_config(self.config_layers,self.H,self.D,self.B,self.theta,ratio,ratio,2.)
        if hold_column is not None:
            for li in range(self.config_layers):
                for h in range(self.H):
                    for kv in (0,1):
                        for key in ['M_angles','M_ratios','M_ratio']:
                            self.cfg[li][h][kv][key]=copy.deepcopy(hold_column[li][h][kv][key])
        self.cloak=KVCloak(self.cfg,dtype=self.cloak_dtype,fused=False,need_ratio=ratio!=1.,add_a=True)
        self.key_seed=key_seed

    def protect(self,cache,seed,cloak=None,capture_evidence=True):
        seed_all(seed)
        # Official operations reuse and overwrite input storage when dtypes match.
        selected=cloak or self.cloak
        if not capture_evidence:return selected.obfuscate(clone_cache(cache))
        from unittest.mock import patch
        permutations=[];original_argsort=torch.argsort
        def record(*a,**kw):
            result=original_argsort(*a,**kw);permutations.append(result.detach().cpu().tolist());return result
        with patch.object(torch,'argsort',record):result=selected.obfuscate(clone_cache(cache))
        if not hasattr(selected,'_evaluation_matrices'):
            prepared=selected._device_cache[result[0][1].device]['obf'][0]
            selected._evaluation_matrices={name:prepared[name+'_v_batch'][0,:,0].float().cpu().clone() for name in ['S','M','A']}
        result.evaluation_only_metadata=dict(permutations=permutations,matrices=selected._evaluation_matrices,fused=selected.fused)
        return result

    def calibrate(self,cloak=None,known_model=None,full_forward=False,capture_evidence=True):
        import math
        target=max(1024,int(2.5*self.B*sum(1/k for k in range(1,self.B+1)))*self.B)
        ids,prefix,char=probe_ids(self.tok,target=target,max_positions=getattr(self.public_model.config,'max_position_embeddings',None))
        ids=ids.to(self.device)
        if full_forward:
            public=self.public_model(input_ids=ids,use_cache=True).past_key_values[0][1][0].float().cpu().clone()
            defended=(known_model or self.model)(input_ids=ids,use_cache=True).past_key_values
        else:
            public=self.cache(ids,model=self.public_model)[0][1][0].float().cpu().clone()
            defended=self.cache(ids,model=known_model or self.model)
        protected=self.protect(defended,1234,cloak,capture_evidence=capture_evidence)
        start=(prefix+self.B-1)//self.B
        count=ids.shape[1]//self.B
        secrets={};rows=[];trace=[]
        fn=legacy_recover if self.calibrator=='legacy' else recover_secrets
        for h in range(self.H):
            blocks=protected[0][1][0,h].float().cpu().reshape(-1,self.B,self.D)[start:count]
            n,centers,intra,inter,labels=cluster_states(blocks)
            item={'head':h,'states':n,'complete':n==self.B,'intra':intra,'inter':inter if math.isfinite(inter) else None}
            evidence=None
            if capture_evidence:
                evidence=dict(head=h,centroids=centers.tolist(),cluster_labels=labels.tolist(),known_public_probe_block=public[h,start*self.B:(start+1)*self.B].tolist(),
                              estimated_parameters=None,effective_protection_parameters_evaluation_only={k:v[h].tolist() for k,v in protected.evaluation_only_metadata['matrices'].items()},
                              declared_configuration_evaluation_only={k:v.tolist() if torch.is_tensor(v) else v for k,v in self.cfg[0][h][1].items()})
                trace.append(evidence)
            if n==self.B:
                sec=fn(centers,public[h,start*self.B:count*self.B],self.B)
                if capture_evidence:item['diagnostics']=ground_truth_errors(sec,self.cfg[0][h][1],public[h,start*self.B:count*self.B])
                if capture_evidence:
                    evidence['estimated_parameters']={k:v.tolist() for k,v in sec.items() if torch.is_tensor(v)}
                secrets[h]={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in sec.items()}
            rows.append(item)
        return secrets,dict(actual_probe_tokens=ids.shape[1],prefix_tokens=prefix,repeat_character=char,heads=rows,
                            probe_input_ids=ids[0].tolist() if capture_evidence else None,
                            recovery_trace_evaluation_only=trace,
                            probe_permutations=protected.evaluation_only_metadata['permutations'] if capture_evidence else None)

    def decode_block(self,C,sec,h,hypothesis_batch=4,return_trace=False):
        best_score=None;best_pred=None;best_trace=None;all_hypotheses=[]
        dictionary=self.normalized[:,h].T
        b=self.B
        for start in range(0,b,hypothesis_batch):
            js=range(start,min(start+hypothesis_batch,b))
            hypotheses=torch.stack([C-torch.outer(sec['s'][j],sec['a_hat']) for j in js])
            X=torch.nn.functional.normalize((sec['s'][None]@hypotheses)@sec['M'].T,dim=-1)
            scores=X@dictionary
            maxima,preds=scores.max(-1)
            sums=maxima.sum(-1)
            if return_trace:all_hypotheses.append(sums.clone())
            j=int(sums.argmax())
            if best_score is None or sums[j].item()>best_score:
                best_score=sums[j].item();best_pred=preds[j].tolist()
                if return_trace:best_trace=dict(selected_j=start+j,normalized_rows=X[j].clone(),row_top_scores=maxima[j].clone())
        if return_trace:
            best_trace['hypothesis_scores']=torch.cat(all_hypotheses)
            return best_pred,best_trace
        return best_pred

    def decode_blocks(self,blocks,sec,h,block_batch=4,hypothesis_batch=4,return_trace=False):
        """The same frozen equation, with multiple independent blocks batched."""
        output=[];traces=[];dictionary=self.normalized[:,h].T
        for offset in range(0,len(blocks),block_batch):
            current=blocks[offset:offset+block_batch];best_scores=None;best_predictions=None
            best_j=None;best_vectors=None;best_rows=None;all_hypotheses=[]
            for start in range(0,self.B,hypothesis_batch):
                js=range(start,min(start+hypothesis_batch,self.B))
                markers=torch.stack([torch.outer(sec['s'][j],sec['a_hat']) for j in js])
                hypotheses=current[:,None]-markers[None]
                X=torch.nn.functional.normalize((sec['s'][None,None]@hypotheses)@sec['M'].T,dim=-1)
                scores=(X.reshape(-1,self.D)@dictionary).reshape(len(current),len(markers),self.B,-1)
                maxima,predictions=scores.max(-1);totals=maxima.sum(-1)
                values,choices=totals.max(-1)
                if return_trace:all_hypotheses.append(totals.clone())
                selected=predictions[torch.arange(len(current),device=self.device),choices]
                if best_scores is None:
                    best_scores=values;best_predictions=selected
                    if return_trace:
                        best_j=choices+start;best_vectors=X[torch.arange(len(current),device=self.device),choices].clone();best_rows=maxima[torch.arange(len(current),device=self.device),choices].clone()
                else:
                    better=values>best_scores
                    best_scores=torch.where(better,values,best_scores)
                    best_predictions=torch.where(better[:,None],selected,best_predictions)
                    if return_trace:
                        best_j=torch.where(better,choices+start,best_j)
                        best_vectors=torch.where(better[:,None,None],X[torch.arange(len(current),device=self.device),choices],best_vectors)
                        best_rows=torch.where(better[:,None],maxima[torch.arange(len(current),device=self.device),choices],best_rows)
            output.extend(best_predictions.tolist())
            if return_trace:
                totals=torch.cat(all_hypotheses,dim=1)
                for i in range(len(current)):traces.append(dict(selected_j=int(best_j[i]),normalized_rows=best_vectors[i],row_top_scores=best_rows[i],hypothesis_scores=totals[i]))
        return (output,traces) if return_trace else output

    def failed_block_diagnostic(self,C,sec,h,truth,prediction,trace,metadata,block):
        """Post-hoc only: ground truth never selects a marker or predicted token."""
        ids=sorted(set(truth));vectors=self.normalized[torch.tensor(ids,device=self.device),h]
        similarities=trace['normalized_rows']@vectors.T
        values,rows=similarities.max(0);tc=Counter(truth);pc=Counter(prediction)
        result=dict(block=block,protected_matrix=C.tolist(),prediction=prediction,truth=truth,selected_j=trace['selected_j'],
                    hypothesis_scores=trace['hypothesis_scores'].tolist(),row_top_scores=trace['row_top_scores'].tolist(),
                    selected_normalized_rows=trace['normalized_rows'].tolist(),
                    posthoc_gold=[dict(token=t,count_true=tc[t],count_predicted=pc[t],absent_from_predictions=pc[t]==0,
                                      count_deficit=max(0,tc[t]-pc[t]),best_score=float(values[i]),best_row=int(rows[i])) for i,t in enumerate(ids)])
        if metadata:
            S=metadata['matrices']['S'][h];A=metadata['matrices']['A'][h]
            matching=torch.nn.functional.cosine_similarity(sec['s'].detach().cpu()[:,None],S.T[None],dim=-1)
            columns=matching.argmax(-1).tolist();permutation=metadata['permutations'][0][block]
            marked=int(A.abs().sum(-1).argmax());column=permutation.index(marked)
            result['evaluation_only_layout']=dict(actual_permutation=permutation,marked_source_row=marked,marked_protection_column=column,
                recovered_column_matching=columns,column_matching_bijection=len(set(columns))==self.B,
                minimum_matched_cosine=float(matching.max(-1).values.min()),expected_j=columns.index(column) if len(set(columns))==self.B else None)
        return result

    def score(self,ids,protected,secrets=None,heads=None):
        secrets=self.secrets if secrets is None else secrets
        heads=range(self.H) if heads is None else heads
        truth=ids[0].tolist();result=[]
        for h in heads:
            bags=[];counts=[];traces=[];diagnostics=[]
            if h in secrets and self.device.type=='cuda':
                blocks=protected[0][1][0,h,:len(truth)].float().reshape(-1,self.B,self.D)
                bags,traces=self.decode_blocks(blocks,secrets[h],h,return_trace=True)
                counts=[sum((Counter(pred)&Counter(truth[k*self.B:(k+1)*self.B])).values()) for k,pred in enumerate(bags)]
            else:
                for k in range(len(truth)//self.B):
                    if h in secrets:
                        pred,trace=self.decode_block(protected[0][1][0,h,k*self.B:(k+1)*self.B].float(),secrets[h],h,return_trace=True);traces.append(trace)
                    else:pred=[]
                    bags.append(pred);counts.append(sum((Counter(pred)&Counter(truth[k*self.B:(k+1)*self.B])).values()))
            if h in secrets:
                for k,c in enumerate(counts):
                    if c!=self.B:diagnostics.append(self.failed_block_diagnostic(protected[0][1][0,h,k*self.B:(k+1)*self.B].float(),secrets[h],h,truth[k*self.B:(k+1)*self.B],bags[k],traces[k],getattr(protected,'evaluation_only_metadata',None),k))
            result.append(dict(head=h,bags=bags,block_correct=counts,correct=sum(counts),total=len(truth),calibration_complete=h in secrets,failed_block_diagnostics_evaluation_only=diagnostics))
        return dict(correct=sum(r['correct'] for r in result),total=sum(r['total'] for r in result),heads=result)

    def verify_decoder(self):
        from attack_lib import attack_block
        ids=self.ids(' '.join(NATURAL))[:,:self.B*4]
        cache=self.cache(ids)
        protected=self.protect(cache,7001)
        checked=[]
        for h in sorted({0,self.H-1}):
            if h not in self.secrets:continue
            blocks=protected[0][1][0,h,:ids.shape[1]].float().reshape(-1,self.B,self.D);sec=self.secrets[h]
            reference=[attack_block(C,sec['s'],sec['a_hat'],sec['M'],self.normalized[:,h]) for C in blocks]
            observed=self.decode_blocks(blocks,sec,h)
            assert observed==reference, 'Batched decoder disagrees with frozen serial reference on the predeclared gate'
            checked.append(dict(head=h,reference=reference,observed=observed))
        return dict(equivalent=True,checks=checked,input_ids=ids[0].tolist())

    def environment(self):
        return dict(model=self.spec,device=str(self.device),hardware=hardware_record(self.device),model_dtype=str(self.dtype),cloak_dtype=str(self.cloak_dtype),
                    source_layers=self.L,configuration_layers=self.config_layers,resident_layers=len(self.model.model.layers),strategy=self.strategy,
                    attention_implementation='eager',calibrator=self.calibrator,block_size=self.B,
                    heads=self.H,head_dimension=self.D,torch=torch.__version__,
                    transformers=__import__('transformers').__version__,tf32=False,torch_threads=torch.get_num_threads(),
                    gate=self.gate,dictionary_validation=self.dictionary_validation,
                    protected_layers=1,cache_extraction='native_cache_update_before_unused_attention',
                    decoder='reference_equation_block_batch4_hypothesis_batch4' if self.device.type=='cuda' else 'reference_equation_single_block_hypothesis_batch4',recovery_dtype='float32',
                    calibration_device='cpu',target_decode_device=str(self.device),
                    cache_forward=self.cache_forward,
                    calibration_executed=hasattr(self,'secrets'),
                    bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
