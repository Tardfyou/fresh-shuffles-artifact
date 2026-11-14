"""Synthetic unit checks only; these do not claim CUDA or model measurements."""
from pathlib import Path
import tempfile,unittest,types,sys,json,copy,ast
from unittest.mock import patch
import torch
from integration.layout import gather,write_page,zero_pages,bits_equal
from integration.bridge import Bridge
from integration import scope_policy
from integration.stores import ByteStore,CloakStore
from integration.records import Records
from get_kvcloak_config import get_kvcloak_config
import exp_veilkv_plus as codec


def caches(block,noncontiguous=False):
    if noncontiguous:return [torch.zeros(2,12,2,block,8,dtype=torch.bfloat16).permute(0,1,3,2,4) for _ in range(2)]
    return [torch.zeros(2,12,block,2,8,dtype=torch.bfloat16) for _ in range(2)]


def identity():return dict(model_dig=b'm'*32,tok_dig=b't'*32,rope=b'r'*32)


def metadata(request,sid,table,tokens):
    return types.SimpleNamespace(request_id=request,seq_data={sid:types.SimpleNamespace(get_token_ids=lambda:list(tokens))},block_tables={sid:table},computed_block_nums=[])


def put_values(c,table,n,block,seed=42):
    torch.manual_seed(seed);plain=tuple((torch.randn(1,2,n,8).to(torch.bfloat16),torch.randn(1,2,n,8).to(torch.bfloat16)) for _ in c)
    for j,page in enumerate(table[:(n+block-1)//block]):write_page(c,page,plain,j,n,block)
    return plain


class Contracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_noncontiguous_real_page_layout(self):
        for block in [16,32]:
            c=caches(block,True);table=[7,2,9];plain=put_values(c,table,block*2+5,block)
            self.assertTrue(bits_equal(plain,gather(c,table,block*2+5,block)))
            zero_pages(c,table);self.assertTrue(all(not bool(x.count_nonzero()) for x in c))
    def test_live_roundtrip_all_valid_rows_and_page_eviction(self):
        for block in [16,32]:
            with tempfile.TemporaryDirectory() as directory:
                c=caches(block,True);b=Bridge(c,block,Path(directory),identity());b.configure('veilkv');scope_policy.register('r','A',0,block);n=block*2+3;table=[7,2,9];plain=put_values(c,table,n,block)
                b.after([metadata('r',1,table,list(range(n)))])
                self.assertTrue(all(not bool(x.count_nonzero()) for x in c));self.assertTrue(b.steps[-1]['roundtrip_bytes_exact'])
                b.before();self.assertTrue(bits_equal(plain,gather(c,table,n,block)));self.assertGreater(b.restored_objects,0)
    def test_growing_partial_page_and_epoch_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            c=caches(16);b=Bridge(c,16,Path(directory),identity());b.configure('veilkv');scope_policy.register('grow','A',0,16);table=[5,3,8]
            for n in [31,32,33,34]:
                if b.refs:b.before()
                plain=put_values(c,table,n,16,n);b.epoch=3 if n<33 else 4;b.after([metadata('grow',1,table,list(range(n)))])
                b.before();self.assertTrue(bits_equal(plain,gather(c,table,n,16)))
            self.assertEqual({s['epoch'] for s in b.steps},{3,4})
    def test_shared_public_physical_page_private_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            c=caches(32);b=Bridge(c,32,Path(directory),identity());b.configure('veilkv');scope_policy.register('warm','P',32,32);prefix=list(range(32));put_values(c,[4],32,32);b.after([metadata('warm',0,[4],prefix)]);b.before()
            scope_policy.register('A','A',32,32);scope_policy.register('B','B',32,32)
            for table,seed in [([4,2],10),([4,7],20)]:
                old=gather(c,[4],32,32);put_values(c,table,48,32,seed);write_page(c,4,old,0,32,32)
            a=prefix+list(range(100,116));d=prefix+list(range(200,216));pa=gather(c,[4,2],48,32);pb=gather(c,[4,7],48,32)
            b.after([metadata('A',1,[4,2],a),metadata('B',2,[4,7],d)]);self.assertGreater(b.byte_store.prefix_reuses,0)
            self.assertEqual(b.payloads('A')['public'],b.payloads('B')['public']);self.assertNotEqual(b.payloads('A')['private'],b.payloads('B')['private'])
            b.before();self.assertTrue(bits_equal(pa,gather(c,[4,2],48,32)));self.assertTrue(bits_equal(pb,gather(c,[4,7],48,32)))
    def test_private_physical_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            c=caches(16);b=Bridge(c,16,Path(directory),identity());b.configure('veilkv')
            for r in ['A','B']:scope_policy.register(r,r,0,16)
            put_values(c,[4,2],32,16)
            with self.assertRaises(AssertionError):b.after([metadata('A',1,[4,2],list(range(32))),metadata('B',2,[4,2],list(range(32)))])
    def test_all_eight_tamper_types_block_restore_and_keep_pages_zero(self):
        for kind in ['ciphertext','tag','principal','nonce','epoch','model','prefix','replay']:
            with tempfile.TemporaryDirectory() as directory:
                c=caches(16);b=Bridge(c,16,Path(directory),identity());b.configure('veilkv');scope_policy.register('bad','A',32,16);table=[4,2,7,8]
                put_values(c,table,49,16);b.after([metadata('bad',1,table,list(range(49)))])
                if kind=='replay':
                    b.before();prefix=gather(c,table,32,16);put_values(c,table,50,16,43)
                    for j,page in enumerate(table[:2]):write_page(c,page,prefix,j,32,16)
                    b.after([metadata('bad',1,table,list(range(50)))])
                b.inject(kind,'bad')
                with self.assertRaises(codec.TamperError):b.before()
                self.assertTrue(all(not bool(x.count_nonzero()) for x in c));self.assertEqual(b.model_calls,0)
    def test_official_cloak_output_stays_fp32_and_ordered_inverse(self):
        torch.manual_seed(42);theta={f'layer_{i}':dict(key_max_values=[1.,1.],value_max_values=[1.,1.]) for i in range(2)};cfg=get_kvcloak_config(2,2,8,16,theta,1.,1.,2.)
        store=CloakStore(cfg,'cpu');plain=tuple((torch.randn(1,2,37,8).to(torch.bfloat16),torch.randn(1,2,37,8).to(torch.bfloat16)) for _ in range(2))
        entry,_=store.seal(plain,[5,2,8],list(range(37)),'r','A',0,3);restored=store.open(entry)
        self.assertTrue(all(x.dtype==torch.float32 for p in entry.packet.tensors for x in p))
        self.assertTrue(all(float((x.float()-y.float()).abs().max())<.001 for p,q in zip(plain,restored) for x,y in zip(p,q)))
        again,_=store.seal(plain,[5,2,8],list(range(37)),'r','A',0,3)
        self.assertTrue(all(torch.equal(x,y) for p,q in zip(entry.packet.tensors,again.packet.tensors) for x,y in zip(p,q)))
    def test_writer_preserves_normal_failures_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Records(directory,'j');r.append('x',functional_pass=False)
            with self.assertRaises(AssertionError):r.append('x',functional_pass=True)
            r.finish({'x'});r.close();self.assertFalse(json.loads((Path(directory)/'records.jsonl').read_text())['functional_pass'])
    def test_scope_hashes_align_public_and_separate_private_before_scheduler(self):
        # Exercise the exact hook interface with host-only API doubles. No vLLM
        # model, CUDA kernel or engine measurement is replaced in delivered runs.
        class Sequence:
            def __init__(self):self.seq_id=1
            def extra_hash(self):return None
        class Scheduler:
            def add_seq_group(self,g):self.added=g
        class Block:
            _none_hash=hash(None)
            @classmethod
            def hash_block_tokens(cls,is_first_block,prev_block_hash,cur_block_token_ids,extra_hash=None):
                if is_first_block and prev_block_hash is None:prev_block_hash=cls._none_hash
                return hash((is_first_block,prev_block_hash,*cur_block_token_ids,extra_hash))
        modules={}
        for n in ['vllm','vllm.core','vllm.core.block','vllm.sequence','vllm.core.scheduler','vllm.core.block.prefix_caching_block']:modules[n]=types.ModuleType(n)
        modules['vllm.sequence'].Sequence=Sequence;modules['vllm.core.scheduler'].Scheduler=Scheduler;modules['vllm.core.block.prefix_caching_block'].PrefixCachingBlock=Block
        old=scope_policy._INSTALLED;scope_policy._INSTALLED=False
        try:
            with patch.dict(sys.modules,modules):
                scope_policy.install();seqs=[]
                for request in ['A','B']:
                    scope_policy.register(request,request,32,16);seq=Sequence();g=types.SimpleNamespace(request_id=request,get_seqs=lambda seq=seq:[seq]);Scheduler().add_seq_group(g);seqs.append(seq)
                def hashes(seq):
                    prev=Block._none_hash;out=[]
                    for i in range(4):prev=Block.hash_block_tokens(i==0,prev,list(range(i*16,(i+1)*16)),seq.extra_hash());out.append(prev)
                    return out
                a,b=map(hashes,seqs);self.assertEqual(a[:2],b[:2]);self.assertNotEqual(a[2:],b[2:])
                self.assertEqual(hashes(copy.deepcopy(seqs[0])),a)
        finally:scope_policy._INSTALLED=old
    def test_registered_counts_and_frozen_input_identity(self):
        from integration.verify import expected
        from integration import ROOT
        for tag in ['qwen25_05b','llama32_1b']:
            rows=[json.loads(x) for x in (ROOT/'data'/(tag+'.jsonl')).read_text().splitlines()]
            self.assertEqual(len(rows),602);self.assertEqual(len({r['sid'] for r in rows}),602)
            self.assertEqual(len(expected(rows,False,False)),6321);self.assertEqual(len(expected(rows,False,True)),12)
            self.assertTrue(all(r['defense_tokens'][:32]==rows[0]['defense_tokens'][:32] for r in rows))
    def test_real_worker_wrapper_authenticates_before_delegate(self):
        import importlib.util
        from integration import ROOT
        class FakeModel:
            def compute_logits(self,hidden,sampling):return torch.arange(19,dtype=torch.float32).reshape(1,-1)
        class FakeRunner:
            def __init__(self,owner):self.owner=owner;self.model=FakeModel()
            def execute_model(self,req):
                self.owner.delegate_calls+=1
                if self.owner.expected_before is not None:
                    assert bits_equal(self.owner.expected_before,gather(self.owner.gpu_cache[0],[4,2,7],37,16)), 'Delegate consumed unrestored pages'
                return 'executed'
        class FakeBase:
            def __init__(self):
                self.parallel_config=types.SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1);self.model_config=types.SimpleNamespace(enforce_eager=True);self.cache_config=types.SimpleNamespace(block_size=16);self.gpu_cache=[caches(16)];self.delegate_calls=0;self.expected_before=None;self.model_runner=FakeRunner(self)
            def execute_model(self,req):
                self.model_runner.execute_model(req)
                n=len(req.seq_group_metadata_list[0].seq_data[1].get_token_ids())
                put_values(self.gpu_cache[0],[4,2,7],n,16,45)
                self.model_runner.model.compute_logits(None,types.SimpleNamespace(seq_groups=[types.SimpleNamespace(seq_ids=[1])]))
                return []
        modules={name:types.ModuleType(name) for name in ['vllm','vllm.worker','vllm.worker.worker']};modules['vllm.worker.worker'].Worker=FakeBase
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules):
            spec=importlib.util.spec_from_file_location('integration._worker_contract',ROOT/'integration/worker.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            worker=module.VeilKVWorker();worker.configure_integration(directory,identity());worker.bridge.configure('veilkv');scope_policy.register('live','A',0,16)
            request=types.SimpleNamespace(virtual_engine=0,num_steps=1,blocks_to_swap_in=[],blocks_to_swap_out=[],blocks_to_copy=[],seq_group_metadata_list=[metadata('live',1,[4,2,7],list(range(37)))])
            worker.execute_model(request);self.assertEqual(worker.delegate_calls,1);self.assertTrue(all(not bool(t.count_nonzero()) for t in worker.gpu_cache[0]))
            worker.bridge.before();worker.expected_before=gather(worker.gpu_cache[0],[4,2,7],37,16);zero_pages(worker.gpu_cache[0],[4,2,7]);worker.execute_model(request);self.assertEqual(worker.delegate_calls,2)
            worker.bridge.inject('ciphertext','live')
            with self.assertRaises(codec.TamperError):worker.execute_model(request)
            self.assertEqual(worker.delegate_calls,2);self.assertEqual(worker.bridge.model_calls,2)

    def test_source_api_nodes_exist(self):
        from integration.api_guard import node_from_file
        from integration import ROOT
        for file,name in [('vllm/core/scheduler.py','Scheduler.add_seq_group'),('vllm/worker/worker_base.py','LocalOrDistributedWorkerBase.execute_model'),('vllm/core/block/prefix_caching_block.py','PrefixCachingBlock.hash_block_tokens')]:self.assertIsInstance(node_from_file(ROOT/'source_reference'/file,name),ast.FunctionDef)


if __name__=='__main__':unittest.main()
