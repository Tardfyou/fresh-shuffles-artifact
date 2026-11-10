"""Validate the exact upstream integration seams before installing hooks."""
import ast,copy,inspect
from pathlib import Path
from . import ROOT

SEAMS=[('vllm/core/scheduler.py','Scheduler.add_seq_group'),('vllm/sequence.py','Sequence.extra_hash'),('vllm/core/block/prefix_caching_block.py','PrefixCachingBlock.hash_block_tokens'),('vllm/worker/worker_base.py','LocalOrDistributedWorkerBase.execute_model'),('vllm/worker/cache_engine.py','CacheEngine._allocate_kv_cache'),('vllm/attention/backends/flash_attn.py','FlashAttentionBackend.get_kv_cache_shape')]


def node_from_file(path,qualname):
    tree=ast.parse(path.read_text());items=tree.body
    for name in qualname.split('.'):
        node=next(n for n in items if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
        items=node.body
    return node


def normalized(node):
    node=copy.deepcopy(node)
    node.decorator_list=[]
    return ast.dump(node,include_attributes=False)


def compare_seam(reference_path,qualname,method):
    """Compare full-file AST nodes without reindenting string constants.

    inspect.getsource returns a method fragment indented within its class.
    Dedenting that fragment changes multiline strings, including docstrings.
    Reading both complete modules preserves those exact constants instead.
    """
    actual_method=inspect.unwrap(method)
    if actual_method.__qualname__!=qualname:
        raise AssertionError('Installed upstream callable differs: '+qualname)
    source=inspect.getsourcefile(actual_method)
    if source is None:
        raise AssertionError('Installed upstream source is unavailable: '+qualname)
    expected=node_from_file(Path(reference_path),qualname)
    actual=node_from_file(Path(source),actual_method.__qualname__)
    if normalized(expected)!=normalized(actual):
        raise AssertionError('Installed upstream seam differs from the issued reference: '+qualname)
    return qualname


def check():
    from vllm.core.scheduler import Scheduler
    from vllm.sequence import Sequence
    from vllm.core.block.prefix_caching_block import PrefixCachingBlock
    from vllm.worker.worker_base import LocalOrDistributedWorkerBase
    from vllm.worker.cache_engine import CacheEngine
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
    checks=[('vllm/core/scheduler.py','Scheduler.add_seq_group',Scheduler.add_seq_group),('vllm/sequence.py','Sequence.extra_hash',Sequence.extra_hash),('vllm/core/block/prefix_caching_block.py','PrefixCachingBlock.hash_block_tokens',PrefixCachingBlock.hash_block_tokens),('vllm/worker/worker_base.py','LocalOrDistributedWorkerBase.execute_model',LocalOrDistributedWorkerBase.execute_model),('vllm/worker/cache_engine.py','CacheEngine._allocate_kv_cache',CacheEngine._allocate_kv_cache),('vllm/attention/backends/flash_attn.py','FlashAttentionBackend.get_kv_cache_shape',FlashAttentionBackend.get_kv_cache_shape)]
    passed=[]
    for file,name,method in checks:
        passed.append(compare_seam(ROOT/'source_reference'/file,name,method))
    return dict(upstream_api_pass=True,compared_functions=passed,
                comparison='Complete-module AST nodes; docstrings and executable string constants preserved')


def check_installed_sources():
    """Preflight source compatibility without importing GPU engine providers."""
    import importlib.metadata,importlib.util
    version=importlib.metadata.version('vllm')
    if version!='0.8.5.post1':raise AssertionError('Use the issued vLLM0.8.5.post1 environment')
    spec=importlib.util.find_spec('vllm')
    if spec is None or spec.origin is None:raise AssertionError('Installed vLLM source is unavailable')
    site_root=Path(spec.origin).parent.parent;passed=[]
    for file,name in SEAMS:
        expected=node_from_file(ROOT/'source_reference'/file,name)
        actual=node_from_file(site_root/file,name)
        if normalized(expected)!=normalized(actual):
            raise AssertionError('Installed upstream source differs from the issued reference: '+name)
        passed.append(name)
    return dict(installed_source_preflight_pass=True,vllm=version,
                compared_functions=passed,model_loaded=False,cuda_initialized=False)


if __name__=='__main__':
    import json
    print(json.dumps(check_installed_sources(),indent=2))
