"""Independent replay of archived model predictions; exits nonzero on mismatch."""
import json,hashlib,platform,sys
import numpy as np
import torch
from run import ROOT,data,contexts,context_menu
from models import make_model
from medmnist import INFO
from analyze import DATASETS,ARCHS
out=[]
for name in DATASETS:
 d=data(name)
 for arch in ARCHS:
  for seed in range(5):
   folder=ROOT/'results'/f'{name}_{arch}_{seed}'
   history=json.loads((folder/'training.json').read_text());assert len(history)==25
   m=make_model(arch,INFO[name]['n_channels'],len(INFO[name]['label']))
   m.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'))
   for mode in (['source'] if arch=='cnn_gn' else ['source','target']):
    path=folder/f'p_clean_{mode}_b16.npz';z=np.load(path)
    order=np.random.default_rng(20406).permutation(len(z['y']))
    assert np.array_equal(z['cal'],order[:len(order)//2]) and np.array_equal(z['test'],order[len(order)//2:])
    expected_contexts,_=context_menu(d['train'][1].numpy(),16)
    assert np.array_equal(z['contexts'],expected_contexts)
    ix=np.array([0,len(z['y'])//3,2*len(z['y'])//3,len(z['y'])-1]);p=contexts(m,d['test'][0][ix],d['train'][0],z['contexts'][:2],mode,'cpu',query_block=2)
    ref=z['p'][:2,ix];err=float(np.max(np.abs(p-ref)));np.testing.assert_allclose(p,ref,atol=5e-5,rtol=5e-4)
    assert np.array_equal(z['y'],d['test'][1].numpy())
    if mode=='source':assert np.array_equal(z['p'],np.broadcast_to(z['p'][0],z['p'].shape))
    out.append(dict(dataset=name,arch=arch,seed=seed,mode=mode,max_cpu_replay_error=err,checkpoint_sha256=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),probability_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print('Verified',len(out),'probability files; maximum CPU replay error',max(r['max_cpu_replay_error'] for r in out))
(ROOT/'results'/'replay_verification.json').write_text(json.dumps(out,indent=2))
