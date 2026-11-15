"""Known-key marker fidelity controls on the existing frozen 446 blocks."""
from collections import Counter
import json
import torch
from transformers.cache_utils import DynamicCache
from . import ROOT
from .studies import harness
from .core import seed_all
from .worker import write
from kvcloak import KVCloak


def run(job,args,store):
    h=harness(job,args,calibration=False)
    seed_all(1234);h.cloak._prepare_device_tensors(h.device)
    no_marker=KVCloak(h.cfg[:1],torch.bfloat16,False,False,False)
    seed_all(1234);no_marker._prepare_device_tensors(h.device)
    parameters=h.cloak._device_cache[h.device]['obf'][0]
    S=parameters['S_v_batch'].reshape(h.H,16,16);M=parameters['M_v_batch'].reshape(h.H,h.D,h.D);A=parameters['A_v_batch'].reshape(h.H,16,h.D)
    rows=[r for r in map(json.loads,(ROOT/'data/marker_numeric_fixtures.jsonl').open()) if r['tag']==job['model']['tag']]
    if job.get('smoke'):rows=rows[:2]
    main={r['sid']:r for r in map(json.loads,(ROOT/'data/samples_main_public.jsonl').open())}
    native_lengths={}
    write(store.root/'environment.json',dict(h.environment(),known_parameters=True,scope='Fixed-candidate vector fidelity, no parameter recovery or marker search',permutation_generator_device=str(h.device)))
    for ordinal,r in enumerate(rows):
        sid='|'.join([r['sid'],str(r['seed']),str(r['h']),str(r['k'])])
        if sid in store.done:continue
        head=r['h'];ids=torch.tensor([r['true']],device=h.device)
        V=h.cache(ids)[0][1][0,head].float()
        if r['sid'] not in native_lengths:native_lengths[r['sid']]=len(h.tok(main[r['sid']]['text']).input_ids)//16
        nb=native_lengths[r['sid']]
        generator=torch.Generator(device=h.device).manual_seed(r['seed'])
        perm=torch.argsort(torch.rand(nb,16,device=h.device,generator=generator),dim=1)[r['k']]
        P=torch.eye(16,device=h.device,dtype=torch.bfloat16)[perm]
        Sp=S[head]@P;invS=torch.linalg.inv(Sp.double());invM=torch.linalg.inv(M[head].double())
        Vm=V.to(torch.bfloat16)@M[head];added=Vm+A[head]
        supplied=torch.zeros(1,h.H,nb*16,h.D,device=h.device,dtype=torch.float32)
        supplied[0,head,r['k']*16:(r['k']+1)*16]=V
        def native(cloak):
            cache=DynamicCache.from_legacy_cache(((torch.zeros_like(supplied),supplied.clone()),))
            seed_all(r['seed'])
            return cloak.obfuscate(cache)[0][1][0,head,r['k']*16:(r['k']+1)*16].double()
        actual=native(h.cloak);without=native(no_marker)
        values={
            'fp64_sanity':(invS@(Sp.double()@(V.double()@M[head].double()+A[head].double()))-A[head].double())@invM,
            'fp32_forward':(invS@(Sp.float()@(V@M[head].float()+A[head].float())).double()-A[head].double())@invM,
            'bf16_known_key':(invS@actual-A[head].double())@invM,
            'bf16_no_marker':(invS@without)@invM,
            'bf16_marker_before_row_mixing':(added.double()-A[head].double())@invM,
            'bf16_no_marker_before_row_mixing':Vm.double()@invM}
        candidate_ids=sorted(set(r['true']+r['variants']['base']['pred']));cand=h.dictionary[torch.tensor(candidate_ids,device=h.device),head].double();cand=torch.nn.functional.normalize(cand,dim=-1)
        true_index=torch.tensor([candidate_ids.index(t) for t in r['true']],device=h.device)
        marker=int(A[head].abs().sum(-1).argmax());trace={}
        for label,value in values.items():
            sims=torch.nn.functional.normalize(value,dim=-1)@cand.T;truth_score=sims[torch.arange(16,device=h.device),true_index]
            other=sims.clone();other[torch.arange(16,device=h.device),true_index]=-torch.inf
            pred=[candidate_ids[i] for i in sims.argmax(-1).tolist()]
            trace[label]=dict(relative_vector_error=((value-V.double()).norm(dim=-1)/V.double().norm(dim=-1).clamp_min(1e-30)).tolist(),predictions=pred,
                true_score=truth_score.tolist(),best_other_score=other.max(-1).values.tolist(),correct_rows=sum(a==b for a,b in zip(pred,r['true'])))
        assert max(trace['fp64_sanity']['relative_vector_error'])<1e-10
        store.append(sid,status='measured',source_sid=r['sid'],head=head,block=r['k'],seed=r['seed'],truth=r['true'],candidate_ids=candidate_ids,
            original_cpu_correct=r['orig_correct'],marker_row=marker,actual_permutation=perm.tolist(),conditions=trace,
            official_forward_used=True,manual_formula_bit_equal=torch.equal(actual.float(),(Sp@added).float()),
            marker_addition_trace=dict(original_value=V[marker].tolist(),bf16_value_after_M=Vm[marker].float().tolist(),stored_marker=A[head,marker].float().tolist(),bf16_sum=added[marker].float().tolist(),
                swallowed_nonzero_coordinates=int(((Vm[marker]!=0)&(added[marker]==A[head,marker])).sum()),nonzero_value_coordinates=int((Vm[marker]!=0).sum())))
    return store.finish(['|'.join([r['sid'],str(r['seed']),str(r['h']),str(r['k'])]) for r in rows])
