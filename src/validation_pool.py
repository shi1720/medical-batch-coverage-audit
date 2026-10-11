"""Seed-0 sensitivity to companions excluded from gradient-based training."""
import argparse,json,time
import numpy as np
import torch
from medmnist import INFO
from run import ROOT,data,contexts,context_menu
from models import make_model
from conformal import calibrate,measure
from analyze import DATASETS,ARCHS,write_csv
parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps');args=parser.parse_args();rows=[]
for name in DATASETS:
 d=data(name);pool=d['val'];ix,names=context_menu(pool[1].numpy(),16)
 for arch in ARCHS[:2]:
  folder=ROOT/'results'/f'{name}_{arch}_0';path=folder/'p_validpool_target_b16.npz'
  if not path.exists():
   m=make_model(arch,INFO[name]['n_channels'],len(INFO[name]['label']));m.load_state_dict(torch.load(folder/'model.pt',weights_only=True));m.to(args.device);p=contexts(m,d['test'][0],pool[0],ix,'target',args.device);m.cpu();del m
   if args.device=='mps':torch.mps.empty_cache()
   y=d['test'][1].numpy();order=np.random.default_rng(20406).permutation(len(y));cal,ev=order[:len(y)//2],order[len(y)//2:]
   np.savez_compressed(path,p=p,y=y,cal=cal,test=ev,contexts=ix,pool='validation')
   (folder/'contexts_validpool_b16.json').write_text(json.dumps({'names':names,'pool':'validation; used for checkpoint selection but not gradient training','batch':16,'unique_counts':[len(np.unique(v)) for v in ix]},indent=2))
  z=np.load(path);p,y,cal,ev=[z[k] for k in ['p','y','cal','test']]
  for method,t in calibrate(p[:,cal],y[cal]).items():rows.append(dict(dataset=name,arch=arch,seed=0,method=method,**measure(p[:,ev],y[ev],t)))
  print('Validation-pool evaluated',name,arch,flush=True)
write_csv(ROOT/'results'/'validation_pool.csv',rows)
