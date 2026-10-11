"""CPU replay of the five 224px model checkpoints and companion menus."""
import hashlib,json
import numpy as np
import torch
from run_highres import OUT,load_data,model,infer
from run import context_menu
from models import ContextBatchNorm2d
rows=[];d=load_data();ix=np.array([0,len(d['test'][1])//3,2*len(d['test'][1])//3,len(d['test'][1])-1])
for seed in range(5):
 folder=OUT/f'seed{seed}';history=json.loads((folder/'training.json').read_text());assert len(history)==5
 m=model(pretrained=False);m.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'))
 # Replay using explicit population moments, independent of the fused MPS path.
 for layer in m.modules():
  if isinstance(layer,ContextBatchNorm2d):layer.fused_target=False
 for mode in ['source','target']:
  path=folder/f'p_clean_{mode}_b16.npz';z=np.load(path);expected,_=context_menu(d['train'][1].numpy(),16);assert np.array_equal(z['contexts'],expected)
  order=np.random.default_rng(20406).permutation(len(z['y']));assert np.array_equal(z['cal'],order[:len(order)//2]);assert np.array_equal(z['test'],order[len(order)//2:]);assert np.array_equal(z['y'],d['test'][1].numpy())
  p=infer(m,d['test'][0][ix],d['train'][0],z['contexts'][:2],mode,'cpu',block=2);ref=z['p'][:2,ix];err=float(np.max(np.abs(p-ref)));np.testing.assert_allclose(p,ref,atol=1e-4,rtol=1e-3)
  rows.append(dict(seed=seed,mode=mode,max_cpu_replay_error=err,checkpoint_sha256=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),probability_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(OUT/'replay_verification.json').write_text(json.dumps(rows,indent=2));print('Verified ten high-resolution probability files, max CPU replay error',max(r['max_cpu_replay_error'] for r in rows))
