"""Public model-loading and threshold utilities only."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(name, threads=16):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    return tok, model

def model_dims(model):
    cfg = model.config
    layers = cfg.num_hidden_layers
    q = getattr(cfg, "num_attention_heads")
    kv = getattr(cfg, "num_key_value_heads", None) or q
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // q
    return layers, kv, head_dim

def build_theta(model, tok, texts):
    """Per-layer/type/head max|cache| over natural texts (mirrors repo theta json)."""
    maxv = {}
    with torch.no_grad():
        for s in texts:
            pv = model(input_ids=tok(s, return_tensors="pt").input_ids, use_cache=True).past_key_values
            for li in range(len(pv)):
                K, V = pv[li]
                e = maxv.setdefault(li, {"key_max_values": torch.full((K.shape[1],), -1.0),
                                         "value_max_values": torch.full((V.shape[1],), -1.0)})
                e["key_max_values"] = torch.maximum(e["key_max_values"], K[0].abs().amax(dim=(1, 2)))
                e["value_max_values"] = torch.maximum(e["value_max_values"], V[0].abs().amax(dim=(1, 2)))
    return {f"layer_{li}": {k: v.tolist() for k, v in d.items()} for li, d in maxv.items()}
