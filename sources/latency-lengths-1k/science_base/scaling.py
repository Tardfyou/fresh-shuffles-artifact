import math
import torch
from attack_lib import rot_strided

def scaled(cents,plain):
 b,d=cents.shape[1:];vs=[]
 for i in range(b):
  for j in range(i+1,b):
   v=torch.linalg.svd(cents[i]-cents[j]).Vh[0]
   if vs and v@vs[0]<0:v=-v
   vs.append(v)
 a=torch.nn.functional.normalize(torch.stack(vs).mean(0),dim=0);a*=1 if a.sum()>0 else -1
 P=torch.eye(d)-torch.outer(a,a);bar=cents.mean(0);u=torch.linalg.svd(bar@P).U[:,0]
 L=(cents-bar)@a
 rows=[];rhs=[]
 for i in range(b):
  for j in range(i+1,b):rows.append([1/b**2,(u@(L[i]+L[j])).item()/b]);rhs.append(-(L[i]@L[j]).item())
 sol=torch.linalg.lstsq(torch.tensor(rows,dtype=torch.float64),torch.tensor(rhs,dtype=torch.float64)).solution
 candidates=[sol[1].item(),math.sqrt(max(0,sol[0].item())),-math.sqrt(max(0,sol[0].item()))];fits=[]
 for q in candidates:
  S=L+q*u[None]/b;uu=S.sum(0);w=(uu@(cents-S[:,:,None]*a)).mean(0)/(uu@uu)
  v=plain[0];v1,v2=v.chunk(2);w1,w2=w.chunk(2)
  angles=torch.atan2(w2,w1)-torch.atan2(v2,v1);ratios=(w1.square()+w2.square()).sqrt()/(v1.square()+v2.square()).sqrt().clamp_min(1e-8)
  gram=S@S.T;off=(gram-torch.diag(gram.diag())).norm()/gram.norm()
  loss=off.item()+math.log((ratios.max()/ratios.min()).item())
  fits.append((loss,dict(s=S,a_hat=a,M=rot_strided(angles,d),ratios=torch.cat([ratios,ratios]),orth_error=off.item(),scale_spread=(ratios.max()/ratios.min()).item())))
 return min(fits,key=lambda x:x[0])[1]
