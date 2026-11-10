"""Regress the real upstream docstring failure and reject meaningful changes."""
import ast,copy,importlib.util,inspect,tempfile,textwrap,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from integration import ROOT
from integration.api_guard import compare_seam,node_from_file,normalized

SEAMS=[('vllm/core/scheduler.py','Scheduler.add_seq_group'),('vllm/sequence.py','Sequence.extra_hash'),('vllm/core/block/prefix_caching_block.py','PrefixCachingBlock.hash_block_tokens'),('vllm/worker/worker_base.py','LocalOrDistributedWorkerBase.execute_model'),('vllm/worker/cache_engine.py','CacheEngine._allocate_kv_cache'),('vllm/attention/backends/flash_attn.py','FlashAttentionBackend.get_kv_cache_shape')]


def callable_from_pinned_ast(path,qualname):
    """Compile the actual selected definition without importing GPU providers.

    The body is never executed. The original source filename and AST constants
    are retained, so Python inspection follows the same path as a loaded method.
    """
    method=copy.deepcopy(node_from_file(path,qualname));method.decorator_list=[]
    cls=ast.ClassDef(name=qualname.split('.')[0],bases=[],keywords=[],body=[method],decorator_list=[])
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.Module(body=[future,cls],type_ignores=[]);ast.fix_missing_locations(module)
    ns={'__name__':'guard_reference_contract'};exec(compile(module,str(path),'exec'),ns)
    return getattr(ns[cls.name],method.name)


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class APIGuardRegression(unittest.TestCase):
    def test_old_guard_fails_on_unchanged_real_sequence_docstring(self):
        path=ROOT/'source_reference/vllm/sequence.py';qualname='Sequence.extra_hash'
        method=callable_from_pinned_ast(path,qualname)
        expected=node_from_file(path,qualname);old_actual=ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        self.assertNotEqual(normalized(expected),normalized(old_actual))
        self.assertNotEqual(ast.get_docstring(expected,clean=False),ast.get_docstring(old_actual,clean=False))
        self.assertEqual(ast.get_docstring(expected),ast.get_docstring(old_actual))
        self.assertEqual(compare_seam(path,qualname,method),qualname)
    def test_all_six_real_pinned_source_definitions_pass(self):
        for file,qualname in SEAMS:
            with self.subTest(seam=qualname):
                path=ROOT/'source_reference'/file;method=callable_from_pinned_ast(path,qualname)
                self.assertEqual(compare_seam(path,qualname,method),qualname)
    def test_rejects_function_body_runtime_strings_and_signature_changes(self):
        original='class Adapter:\n    def route(self, value=3):\n        """Keep this documentation.\n\n        This line retains its original indentation.\n        """\n        namespace = "private-tail"\n        return value + 1, namespace\n'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference.py';reference.write_text(original)
            variants={'same':original,'body':original.replace('value + 1','value + 2'),'string':original.replace('"private-tail"','"public-prefix"'),'signature':original.replace('value=3','value=4'),'docstring':original.replace('Keep this documentation.','Different documentation.')}
            for name,source in variants.items():
                with self.subTest(variant=name):
                    path=root/(name+'.py');path.write_text(source);method=load_module(path,'guard_'+name).Adapter.route
                    if name=='same':self.assertEqual(compare_seam(reference,'Adapter.route',method),'Adapter.route')
                    else:
                        with self.assertRaises(AssertionError):compare_seam(reference,'Adapter.route',method)
    def test_rejects_callable_with_different_qualified_name(self):
        path=ROOT/'source_reference/vllm/sequence.py';method=callable_from_pinned_ast(path,'Sequence.extra_hash');method.__qualname__='Other.extra_hash'
        with self.assertRaises(AssertionError):compare_seam(path,'Sequence.extra_hash',method)
    def test_installed_source_preflight_without_importing_engine(self):
        import sys
        from integration.api_guard import check_installed_sources
        before=sys.modules.get('vllm')
        origin=ROOT/'source_reference/vllm/__init__.py'
        with patch('importlib.metadata.version',return_value='0.8.5.post1'),patch('importlib.util.find_spec',return_value=SimpleNamespace(origin=str(origin))):
            report=check_installed_sources()
        self.assertTrue(report['installed_source_preflight_pass']);self.assertEqual(len(report['compared_functions']),6)
        self.assertFalse(report['model_loaded']);self.assertFalse(report['cuda_initialized'])
        self.assertIs(sys.modules.get('vllm'),before)
    def test_installed_source_preflight_rejects_wrong_version(self):
        from integration.api_guard import check_installed_sources
        with patch('importlib.metadata.version',return_value='0.9.0'):
            with self.assertRaises(AssertionError):check_installed_sources()

    def test_normalization_does_not_mutate_input_ast(self):
        path=ROOT/'source_reference/vllm/core/block/prefix_caching_block.py';node=node_from_file(path,'PrefixCachingBlock.hash_block_tokens');before=ast.dump(node,include_attributes=True)
        normalized(node);self.assertEqual(ast.dump(node,include_attributes=True),before)


if __name__=='__main__':unittest.main()
