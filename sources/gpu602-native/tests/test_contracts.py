"""Scientific conservation, independent-reference and writer-safety checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache

from runner.core import Harness,seed_all
from runner.records import Records
from runner import lock
from attack_lib import attack_block
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak


class ScientificContracts(unittest.TestCase):
    def test_byte_codec_large_head_geometry_and_bf16_bits(self):
        import exp_veilkv_plus as codec
        from corrected.freshness import KeyService
        with tempfile.TemporaryDirectory() as directory:
            seed_all(42)
            source=tuple((torch.randn(1,8,32,128).to(torch.bfloat16),torch.randn(1,8,32,128).to(torch.bfloat16)) for _ in range(2))
            ctx=dict(model_dig=b'm'*32,tok_dig=b't'*32,rope=b'r'*32,tokens=torch.arange(32).reshape(1,-1),
                     n_prefix_blocks=0,profile='tee',storage_dtype=torch.bfloat16,principal=b'tenant',nonce=b'n'*16,
                     epoch_id=3,variant={},bsz=1,H=8,D=128,true_len=32,key_service=KeyService(Path(directory)/'freshness.sqlite'))
            restored=codec.recover_cache(codec.protect_cache(source,ctx),ctx)
            self.assertTrue(all(torch.equal(a.view(torch.uint8),b.to(torch.bfloat16).view(torch.uint8)) for p,q in zip(source,restored) for a,b in zip(p,q)))

    def test_private_only_identity_binding_and_stream_separation(self):
        import exp_veilkv_plus as codec
        from corrected.freshness import KeyService
        with tempfile.TemporaryDirectory() as directory:
            ctx=dict(model_dig=b'm'*32,tok_dig=b't'*32,rope=b'r'*32,tokens=torch.arange(32).reshape(1,-1),
                     n_prefix_blocks=0,profile='tee',storage_dtype=torch.float32,principal=b'tenant',nonce=b'n'*16,
                     epoch_id=3,variant={},bsz=1,H=2,D=64,true_len=32,key_service=KeyService(Path(directory)/'freshness.sqlite'))
            source=((torch.randn(1,2,32,64),torch.randn(1,2,32,64)),)
            obj=codec.protect_cache(source,ctx)
            restored=codec.recover_cache(obj,ctx)
            self.assertTrue(all(torch.equal(a,b) for p,q in zip(source,restored) for a,b in zip(p,q)))
            for field in ['model_dig','tok_dig','rope']:
                with self.assertRaises(codec.TamperError):codec.recover_cache(obj,{**ctx,field:b'x'*32})
            changed={**ctx,'model_dig':b'x'*32}
            other=codec.protect_cache(source,changed)
            self.assertNotEqual(obj.blocks[(0,0)][0]['ct'],other.blocks[(0,0)][0]['ct'])
            obj.format_version=1
            with self.assertRaises(codec.TamperError):codec.recover_cache(obj,ctx)

    def test_model_identity_binds_revision_not_local_path(self):
        from types import SimpleNamespace
        from runner.identity import checkpoint_identity
        class Tokenizer:
            special_tokens_map={'eos_token':'EOS'}
            padding_side='right';truncation_side='right'
            def __len__(self):return 257
        model=SimpleNamespace(config=SimpleNamespace(model_type='test',rope_theta=10000.,rope_scaling=None))
        tok=Tokenizer();tok.init_kwargs={'name_or_path':'/machine-one/cache'}
        spec=dict(repo='public/model',revision='a'*40)
        a=checkpoint_identity(model,tok,spec)
        tok.init_kwargs={'name_or_path':'/another-machine/cache'}
        self.assertEqual(a,checkpoint_identity(model,tok,spec))
        b=checkpoint_identity(model,tok,{**spec,'revision':'b'*40})
        self.assertNotEqual(a['model_dig'],b['model_dig'])
        model.config.rope_scaling={'factor':2.}
        self.assertNotEqual(a['rope'],checkpoint_identity(model,tok,spec)['rope'])

    def test_batched_equation_against_frozen_serial_reference(self):
        torch.set_num_threads(2)
        for b,d in [(8,64),(16,64),(16,128),(32,128)]:
            seed_all(123)
            h=object.__new__(Harness);h.B=b;h.D=d;h.H=1;h.device=torch.device('cpu')
            h.normalized=torch.nn.functional.normalize(torch.randn(257,1,d),dim=-1)
            sec=dict(s=torch.linalg.qr(torch.randn(b,b)).Q,a_hat=torch.randn(d),M=torch.linalg.qr(torch.randn(d,d)).Q)
            blocks=torch.randn(5,b,d)
            expected=[attack_block(c,sec['s'],sec['a_hat'],sec['M'],h.normalized[:,0]) for c in blocks]
            self.assertEqual(h.decode_blocks(blocks,sec,0),expected)

    def test_bf16_api_inplace_mutation_is_contained(self):
        seed_all(42)
        theta={f'layer_{i}':{'key_max_values':[1.,1.],'value_max_values':[1.,1.]} for i in range(2)}
        cfg=get_kvcloak_config(2,2,64,16,theta,1.,1.,2.)
        h=object.__new__(Harness);h.cloak=KVCloak(cfg,torch.bfloat16,False,False,True)
        seed_all(1234);h.cloak._prepare_device_tensors(torch.device('cpu'))
        source=DynamicCache.from_legacy_cache(tuple((torch.randn(1,2,32,64).to(torch.bfloat16),torch.randn(1,2,32,64).to(torch.bfloat16)) for _ in range(2)))
        before=tuple((a.clone(),b.clone()) for a,b in source)
        h.protect(source,5001);a=h.protect(source,5002);b=h.protect(source,5002)
        self.assertTrue(all(torch.equal(x,y) for p,q in zip(before,source) for x,y in zip(p,q)))
        self.assertTrue(all(torch.equal(x,y) for p,q in zip(a,b) for x,y in zip(p,q)))

    def test_wrong_outcomes_are_complete_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Records(directory,dict(id='test'))
            r.append('negative',status='measured',functional_pass=False)
            self.assertIn('negative',r.done)
            with self.assertRaises(AssertionError):r.append('negative',status='measured',functional_pass=True)
            r.finish({'negative'});r.close()
            self.assertFalse(json.loads((Path(directory)/'records.jsonl').read_text())['functional_pass'])

    def test_two_writers_cannot_own_the_same_job(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Records(directory,dict(id='test'))
            with self.assertRaises(RuntimeError):Records(directory,dict(id='test'))
            r.close()

    def test_missing_ids_never_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Records(directory,dict(id='test'));r.append('a',status='measured')
            with self.assertRaises(AssertionError):r.finish({'a','b'})
            r.close()

    def test_code_lock_detects_scientific_file_edits(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'science.py';p.write_text('x=1\n')
            with zipfile.ZipFile(root/'baseline.zip','w') as z:z.writestr('science.py',p.read_bytes())
            (root/'SCIENTIFIC_LOCK.json').write_text(json.dumps(dict(version='test',baseline_archive='baseline.zip',files=['science.py'])))
            with patch.object(lock,'ROOT',root):
                self.assertTrue(lock.verify()['pass_lock']);p.write_text('x=2\n')
                with self.assertRaises(RuntimeError):lock.verify()


if __name__=='__main__':unittest.main()
