"""Source-aligned MMLU/SQuAD block-size controls with frozen complete cohorts."""
import argparse
import json
from pathlib import Path
import random
import re
import string

import torch

from . import ROOT
from .core import clone_cache,seed_all,stable_seed,synchronize
from .full_model import Consumer
from .worker import write
from get_kvcloak_config import get_kvcloak_config
from kvcloak import KVCloak
from run_suite import NATURAL


def prepare():
    from datasets import load_dataset
    specs={r['repo']:r['revision'] for r in json.loads((ROOT/'protocol/utility_datasets.json').read_text())}
    dest=ROOT/'data/utility';dest.mkdir(exist_ok=True)
    for name in ['mmlu','squad']:
        if (dest/(name+'.jsonl')).exists():continue
        if name=='mmlu':
            ds=load_dataset('cais/mmlu','all',revision=specs['cais/mmlu'])
            examples={}
            for subject in sorted(ds['test'].unique('subject')):
                dev=ds['dev'].filter(lambda r:r['subject']==subject).shuffle(seed=42)
                assert len(dev)>=5
                examples[subject]=list(dev.select(range(5)))
            rows=[]
            for i,r in enumerate(ds['test']):
                subject=r['subject'];prompt='The following are multiple choice questions (with answers) about '+ ' '.join(w.capitalize() for w in subject.split('_'))+'.\n\n'
                for ex in examples[subject]:
                    prompt+='Question: '+ex['question']+'\n'
                    for j,choice in enumerate(ex['choices']):prompt+='ABCD'[j]+'. '+choice+'\n'
                    prompt+='Answer: '+'ABCD'[ex['answer']]+'\n\n'
                prompt+='Question: '+r['question']+'\n'
                for j,choice in enumerate(r['choices']):prompt+='ABCD'[j]+'. '+choice+'\n'
                prompt+='Answer'
                rows.append(dict(sid='mmlu-'+str(i),subject=subject,prompt=prompt,truth='ABCD'[r['answer']]))
            assert len(rows)==14042
        else:
            ds=load_dataset('rajpurkar/squad',revision=specs['rajpurkar/squad']);rng=random.Random(42);rows=[]
            for r in ds['validation']:
                idx=rng.randrange(len(ds['train']));ex=ds['train'][idx]
                prompt=f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answers']['text'][0]}\n\nContext: {r['context']}\nQuestion: {r['question']}\nAnswer"
                rows.append(dict(sid='squad-'+r['id'],prompt=prompt,truths=r['answers']['text'],one_shot_train_index=idx))
            assert len(rows)==10570
        with (dest/(name+'.jsonl')).open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        (dest/(name+'-protocol.json')).write_text(json.dumps(dict(dataset=name,rows=len(rows),source='Official prompt/metric definitions; every test item retained.',python_example_seed=42,example_selection_frozen_across_conditions=True,original_unseeded_squad_choice_replaced_with_declared_seed=name=='squad'),indent=2)+'\n')
        print(name,len(rows),'frozen',flush=True)


def normalize(text):
    text=re.sub(r'\b(a|an|the)\b',' ',text.lower())
    return ' '.join(''.join(ch for ch in text if ch not in set(string.punctuation)).split())


def run(job,args,store):
    c=Consumer(job,args);path=ROOT/'data/utility'/(job['benchmark']+'.jsonl')
    if not path.exists():raise RuntimeError('Prepare frozen utility datasets first: python -m runner.utility --prepare')
    rows=[json.loads(line) for line in path.open()]
    if job.get('smoke'):rows=rows[:2]
    cloak=None
    if job['protection']=='kvcloak':
        maxima={}
        for text in NATURAL:
            cache=c.forward(c.tokenizer(text).input_ids,keep=1).past_key_values
            for li,(K,V) in enumerate(cache):
                d={'key_max_values':K[0].abs().amax(dim=(1,2)).float().cpu(),'value_max_values':V[0].abs().amax(dim=(1,2)).float().cpu()}
                maxima[li]=d if li not in maxima else {k:torch.maximum(maxima[li][k],v) for k,v in d.items()}
        theta={f'layer_{li}':{k:v.tolist() for k,v in d.items()} for li,d in maxima.items()}
        seed_all(42);cfg=get_kvcloak_config(c.L,c.H,c.D,job['block'],theta,1.,1.,2.)
        cloak=KVCloak(cfg,c.dtype,False,False,True)
    write(store.root/'environment.json',dict(c.metadata(),benchmark=job['benchmark'],protection=job['protection'],block_size=job['block'],prompt_source='Frozen upstream-format MMLU5shot/SQuAD1shot',error_policy='No skipped questions'))
    newline=c.tokenizer.encode('\n\n',add_special_tokens=False)[-1]
    stop={c.tokenizer.eos_token_id,newline}
    trigger=c.tokenizer.encode(':',add_special_tokens=False)
    for row in rows:
        if row['sid'] in store.done:continue
        inputs=c.tokenizer(row['prompt'],**(dict(max_length=1024,truncation=True) if job['benchmark']=='squad' else {})).input_ids
        out=c.forward(inputs,keep=1);cache=out.past_key_values
        if cloak:
            seed_all(stable_seed(row['sid']));cache=cloak.deobfuscate(cloak.obfuscate(clone_cache(cache)))
        next_input=trigger;generated=[]
        for _ in range(1 if job['benchmark']=='mmlu' else 64):
            out=c.forward(next_input,cache=cache,keep=1);token=int(out.logits[0,-1].argmax())
            if job['benchmark']=='squad' and token in stop:break
            generated.append(token);cache=out.past_key_values;next_input=[token]
        text=c.tokenizer.decode(generated,skip_special_tokens=True).strip()
        if job['benchmark']=='mmlu':
            upper=text.upper();prediction=upper[0] if upper and upper[0] in 'ABCD' else 'Z';correct=prediction==row['truth']
        else:prediction=text;correct=any(normalize(text)==normalize(t) for t in row['truths'])
        store.append(row['sid'],status='measured',input_ids=inputs,prediction=prediction,prediction_ids=generated,correct=correct,
                     truth=row.get('truth'),truths=row.get('truths'),subject=row.get('subject'))
    return store.finish([r['sid'] for r in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
