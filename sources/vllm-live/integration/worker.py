"""Pinned vLLM worker extension around actual cache consumption and production."""
import torch
from vllm.worker.worker import Worker
from .bridge import Bridge


class VeilKVWorker(Worker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.bridge=None;self.current_request_by_sequence={};self.logit_frames=[]
    def configure_integration(self,directory,identity):
        assert self.parallel_config.tensor_parallel_size==1 and self.parallel_config.pipeline_parallel_size==1
        assert self.model_config.enforce_eager
        self.bridge=Bridge(self.gpu_cache[0],self.cache_config.block_size,__import__('pathlib').Path(directory),identity)
        original=self.model_runner.execute_model
        def execute(*args,**kwargs):
            self.bridge.model_calls+=1;return original(*args,**kwargs)
        self.model_runner.execute_model=execute
        model=self.model_runner.model;compute=model.compute_logits
        def logits(hidden_states,sampling_metadata):
            value=compute(hidden_states,sampling_metadata)
            if value is not None:
                sequence_ids=[sid for group in sampling_metadata.seq_groups for sid in group.seq_ids]
                assert len(sequence_ids)==value.shape[0],'Unexpected selected-logit row layout'
                self.logit_frames.append({self.current_request_by_sequence[sid]:row.detach().float().cpu().clone() for sid,row in zip(sequence_ids,value)})
            return value
        model.compute_logits=logits
        return self.bridge.layout
    @torch.inference_mode()
    def execute_model(self,execute_model_req=None):
        if self.bridge is None or execute_model_req is None:return super().execute_model(execute_model_req)
        assert execute_model_req.virtual_engine==0 and execute_model_req.num_steps==1
        assert not execute_model_req.blocks_to_swap_in and not execute_model_req.blocks_to_swap_out and not execute_model_req.blocks_to_copy,'This registered greedy/no-swap queue must not silently enter untested cache movement'
        metadata=execute_model_req.seq_group_metadata_list
        assert all(hasattr(group,'seq_data') for group in metadata),'Use full V0 metadata, not an unadapted delta path'
        self.current_request_by_sequence={sid:group.request_id for group in metadata for sid in group.seq_data}
        self.bridge.before()
        try:result=super().execute_model(execute_model_req)
        except Exception:
            from .layout import zero_pages
            if self.bridge.mode!='plain':zero_pages(self.bridge.caches,list(self.bridge.refs))
            raise
        self.bridge.after(metadata)
        return result
