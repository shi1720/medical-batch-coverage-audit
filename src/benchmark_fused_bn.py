"""Numerical parity and runtime check only; no performance-based research selection."""
import json,time
import numpy as np
import torch
from run_highres import OUT,model,prepare,load_data,infer
from models import ContextBatchNorm2d
from run import ROOT
m=model(pretrained=False);m.load_state_dict(torch.load(OUT/'seed0'/'model.pt',weights_only=True));m.to('mps');d=load_data();z=np.load(OUT/'seed0'/'p_clean_target_b16.npz');x=d['test'][0][:16];ix=z['contexts'][:1]
results={};preds={}
for fast in [False,True]:
 for layer in m.modules():
  if isinstance(layer,ContextBatchNorm2d):layer.fused_target=fast
 # Warm up compiled shapes, then time two complete context evaluations.
 infer(m,x,d['train'][0],ix,'target','mps');times=[]
 for _ in range(2):
  start=time.time();p=infer(m,x,d['train'][0],ix,'target','mps');times.append(time.time()-start)
 results[str(fast)]=times;preds[str(fast)]=p
err=float(np.max(np.abs(preds['True']-preds['False'])));np.testing.assert_allclose(preds['True'],preds['False'],atol=2e-5,rtol=2e-4)
r=dict(seconds=results,max_probability_difference=err,queries=16,contexts=1,block=4,concurrent_other_processes=True)
(ROOT/'fused_bn_benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
