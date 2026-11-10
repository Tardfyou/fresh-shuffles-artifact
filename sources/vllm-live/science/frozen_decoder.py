"""Frozen decoding methods from the reviewed GPU bundle."""
import torch

class Decoder:
    def __init__(self,dictionary,block=16):
        self.B=block;self.D=dictionary.shape[-1];self.H=dictionary.shape[1];self.device=dictionary.device
        self.normalized=torch.nn.functional.normalize(dictionary,dim=-1)

    def decode_block(self,C,sec,h,hypothesis_batch=4,return_trace=False):
        best_score=None;best_pred=None;best_trace=None;all_hypotheses=[]
        dictionary=self.normalized[:,h].T
        b=self.B
        for start in range(0,b,hypothesis_batch):
            js=range(start,min(start+hypothesis_batch,b))
            hypotheses=torch.stack([C-torch.outer(sec['s'][j],sec['a_hat']) for j in js])
            X=torch.nn.functional.normalize((sec['s'][None]@hypotheses)@sec['M'].T,dim=-1)
            scores=X@dictionary
            maxima,preds=scores.max(-1)
            sums=maxima.sum(-1)
            if return_trace:all_hypotheses.append(sums.clone())
            j=int(sums.argmax())
            if best_score is None or sums[j].item()>best_score:
                best_score=sums[j].item();best_pred=preds[j].tolist()
                if return_trace:best_trace=dict(selected_j=start+j,normalized_rows=X[j].clone(),row_top_scores=maxima[j].clone())
        if return_trace:
            best_trace['hypothesis_scores']=torch.cat(all_hypotheses)
            return best_pred,best_trace
        return best_pred

    def decode_blocks(self,blocks,sec,h,block_batch=4,hypothesis_batch=4,return_trace=False):
        """The same frozen equation, with multiple independent blocks batched."""
        output=[];traces=[];dictionary=self.normalized[:,h].T
        for offset in range(0,len(blocks),block_batch):
            current=blocks[offset:offset+block_batch];best_scores=None;best_predictions=None
            best_j=None;best_vectors=None;best_rows=None;all_hypotheses=[]
            for start in range(0,self.B,hypothesis_batch):
                js=range(start,min(start+hypothesis_batch,self.B))
                markers=torch.stack([torch.outer(sec['s'][j],sec['a_hat']) for j in js])
                hypotheses=current[:,None]-markers[None]
                X=torch.nn.functional.normalize((sec['s'][None,None]@hypotheses)@sec['M'].T,dim=-1)
                scores=(X.reshape(-1,self.D)@dictionary).reshape(len(current),len(markers),self.B,-1)
                maxima,predictions=scores.max(-1);totals=maxima.sum(-1)
                values,choices=totals.max(-1)
                if return_trace:all_hypotheses.append(totals.clone())
                selected=predictions[torch.arange(len(current),device=self.device),choices]
                if best_scores is None:
                    best_scores=values;best_predictions=selected
                    if return_trace:
                        best_j=choices+start;best_vectors=X[torch.arange(len(current),device=self.device),choices].clone();best_rows=maxima[torch.arange(len(current),device=self.device),choices].clone()
                else:
                    better=values>best_scores
                    best_scores=torch.where(better,values,best_scores)
                    best_predictions=torch.where(better[:,None],selected,best_predictions)
                    if return_trace:
                        best_j=torch.where(better,choices+start,best_j)
                        best_vectors=torch.where(better[:,None,None],X[torch.arange(len(current),device=self.device),choices],best_vectors)
                        best_rows=torch.where(better[:,None],maxima[torch.arange(len(current),device=self.device),choices],best_rows)
            output.extend(best_predictions.tolist())
            if return_trace:
                totals=torch.cat(all_hypotheses,dim=1)
                for i in range(len(current)):traces.append(dict(selected_j=int(best_j[i]),normalized_rows=best_vectors[i],row_top_scores=best_rows[i],hypothesis_scores=totals[i]))
        return (output,traces) if return_trace else output
