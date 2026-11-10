"""Reuse storage/resource settings in v2 while leaving all v1 files untouched."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('old_package',type=Path);a=p.parse_args();old=a.old_package.expanduser().resolve()
assert old!=ROOT.resolve() and (old/'runtime/settings.json').is_file()
old_lock=json.loads((old/'SCIENTIFIC_LOCK.json').read_text())
assert old_lock['version']=='vllm-live-085post1-20261110-v1','Point to the original v1 package directory'
new_lock=json.loads((ROOT/'SCIENTIFIC_LOCK.json').read_text())
assert new_lock['version']=='vllm-live-085post1-20261110-v2'
settings=json.loads((old/'runtime/settings.json').read_text());old_output=Path(settings['output_dir']).expanduser();old_output=old_output if old_output.is_absolute() else old/old_output
cache=Path(settings['cache_dir']).expanduser();cache=cache if cache.is_absolute() else old/cache
settings['cache_dir']=str(cache.resolve());settings['output_dir']='results'
new_output=ROOT/'results';assert new_output.resolve()!=old_output.resolve()
existing=new_output/'SMOKE_GATE.json'
if existing.exists():assert json.loads(existing.read_text())['source_lock']['version']==new_lock['version']
(ROOT/'runtime/settings.json').write_text(json.dumps(settings,indent=2)+'\n')
(ROOT/'runtime/V1_REUSE.json').write_text(json.dumps(dict(v1_package=str(old),v1_output_preserved=str(old_output.resolve()),v2_output=str(new_output),model_cache_reused=True,old_files_modified=False,old_measurements_copied=False),indent=2)+'\n')
print('Reused GPU/resource settings and model cache. v1 files are untouched; v2 uses its own results directory.')
