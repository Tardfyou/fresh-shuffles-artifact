"""Score retained CUDA ordering outputs with the same fixed sentence encoder."""
import argparse
import json
import os
from pathlib import Path
from summarize_gpu_results import records


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path);a=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if a.cache:os.environ['HF_HOME']=str(a.cache)
    import torch
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    rows=sorted(records(a.records),key=lambda r:r['sid']);assert len(rows)==len({r['sid'] for r in rows})==602
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B',revision='060db6499f32faf8b98477b0a26969ef7d8b9987',local_files_only=True)
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');assert encoder.max_seq_length==256
    ref=encoder.encode([tok.decode(r['truth'],skip_special_tokens=False) for r in rows],batch_size=32,convert_to_tensor=True,show_progress_bar=False)
    output=[dict(sid=r['sid'],conditions={}) for r in rows]
    for key in ['end2end','oracle_ms','no_cross_block','random_order']:
        texts=[tok.decode(r['conditions'][key]['prediction'],skip_special_tokens=False) for r in rows]
        vec=encoder.encode(texts,batch_size=32,convert_to_tensor=True,show_progress_bar=False)
        for r,v in zip(output,torch.nn.functional.cosine_similarity(ref,vec).tolist()):r['conditions'][key]=v
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(r)+'\n' for r in output));print(a.output)


if __name__=='__main__':main()
