"""Read-only GPU/software preflight plus a tiny native arithmetic check."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from . import ROOT
from .paths import result_root


def inspect(gpu=0,smoke=True):
    import torch
    import transformers
    report=dict(python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
                transformers=transformers.__version__,platform=platform.platform(),
                cuda_available=torch.cuda.is_available(),working_directory=str(ROOT))
    if not report['cuda_available']:raise RuntimeError('CUDA-enabled PyTorch and an accessible GPU are required')
    torch.cuda.set_device(gpu)
    props=torch.cuda.get_device_properties(gpu)
    free,total=torch.cuda.mem_get_info(gpu)
    if total<78*(1<<30):raise RuntimeError('Select the 80-GiB GPU for this complete queue')
    report['gpu']=dict(index=gpu,name=props.name,total_bytes=total,free_bytes=free,compute_capability=[props.major,props.minor],bf16_supported=torch.cuda.is_bf16_supported())
    report['disk_free_bytes']=shutil.disk_usage(ROOT).free
    report['versions_match_native_reference']=torch.__version__.split('+')[0]=='2.6.0' and transformers.__version__=='4.51.3'
    if not report['versions_match_native_reference']:
        raise RuntimeError('Use the pinned Torch2.6.0/Transformers4.51.3 native environment')
    if smoke:
        if free<1<<30:raise RuntimeError('Wait for at least one GiB free before the tiny CUDA check')
        for dtype in [torch.float32,torch.bfloat16]:
            x=torch.ones((256,256),dtype=dtype,device=f'cuda:{gpu}');out=x@x;torch.cuda.synchronize(gpu)
            assert bool((out==256).all()) and bool(torch.isfinite(out).all())
            report[str(dtype)]=dict(tiny_matmul_pass=True,value=float(out[0,0]))
            del x,out
        torch.cuda.empty_cache()
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        report['nvidia_smi']=p.stdout.strip()
    except Exception as e:report['nvidia_smi_error']=str(e)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--output',type=Path,default=result_root()/'DOCTOR.json');a=p.parse_args()
    result=inspect(a.gpu);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
