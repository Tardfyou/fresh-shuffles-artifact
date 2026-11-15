"""Stable trusted metadata labels, independent of machine-local cache paths."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,separators=(',',':')).encode()).digest()


def checkpoint_identity(model,tokenizer,spec):
    assert spec.get('repo') and isinstance(spec.get('revision'),str) and len(spec['revision'])==40
    model_label=dict(domain='checkpoint-model-v1',repo=spec['repo'],revision=spec['revision'],
                     model_type=getattr(model.config,'model_type',None))
    tokenizer_label=dict(domain='checkpoint-tokenizer-v1',repo=spec['repo'],revision=spec['revision'],
                         tokenizer_class=type(tokenizer).__name__,vocabulary_size=len(tokenizer),
                         special_tokens=tokenizer.special_tokens_map,
                         padding_side=tokenizer.padding_side,truncation_side=tokenizer.truncation_side)
    positional={name:getattr(model.config,name,None) for name in ['rope_theta','rope_scaling','max_position_embeddings','sliding_window']}
    return dict(model_dig=digest(model_label),tok_dig=digest(tokenizer_label),rope=digest(positional))
