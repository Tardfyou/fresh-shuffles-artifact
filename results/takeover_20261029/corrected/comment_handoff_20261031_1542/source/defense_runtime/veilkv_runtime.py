"""Trusted model-runtime adaptation after authenticated VeilKV recovery."""
import torch
import exp_veilkv_plus as codec


def recover_for_model(protected, context, *, model_dtype):
    """Authenticate with the original codec, then restore the consumer dtype.

    The original codec returns FP32 tensors even for bf16 storage. Its default
    API and stored representation remain unchanged. The trusted caller supplies
    the model dtype separately from the authenticated storage dtype.
    """
    if model_dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError('Validated consumer dtypes are float32 and bfloat16')
    cache=codec.recover_cache(protected,context)
    return tuple((key.to(dtype=model_dtype),value.to(dtype=model_dtype)) for key,value in cache)
