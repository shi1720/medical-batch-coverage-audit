"""Deterministic cumulative-probability (APS) sensitivity; primary score stays 1-p."""
import numpy as np
from analyze import ROOT,DATASETS,ARCHS,write_csv
from conformal import calibrate,measure

def aps_scores(p,uniform=None):
 p=np.asarray(p,dtype=np.float64)
 p=p/p.sum(axis=-1,keepdims=True)
 order=np.argsort(-p,axis=-1,kind='stable')
 cumulative=np.cumsum(np.take_along_axis(p,order,axis=-1),axis=-1)
 cumulative=np.clip(cumulative,0,1);cumulative[...,-1]=1.
 scores=np.take_along_axis(cumulative,np.argsort(order,axis=-1),axis=-1)
 if uniform is not None:
  u=np.asarray(uniform)
  if u.shape != (p.shape[1],) or np.any((u<0)|(u>1)):raise ValueError('one uniform in [0,1] per query required')
  scores=scores-u[None,:,None]*p
 return scores

if __name__=='__main__':
 rows=[]
 for d in DATASETS:
  for a in ARCHS[:2]:
   for seed in range(5):
    path=ROOT/'results'/f'{d}_{a}_{seed}'/'p_clean_target_b16.npz'
    if not path.exists():continue
    z=np.load(path);p,y,cal,ev=[z[k] for k in ['p','y','cal','test']]
    uniform=np.random.default_rng(90210).random(len(y))
    for score,scores in [('deterministic_APS',aps_scores(p)),('randomized_APS',aps_scores(p,uniform))]:
     for method,t in calibrate(p[:,cal],y[cal],nonconformity=scores[:,cal]).items():
      rows.append(dict(dataset=d,arch=a,seed=seed,score=score,method=method,**measure(p[:,ev],y[ev],t,nonconformity=scores[:,ev])))
 write_csv(ROOT/'results'/'score_sensitivity.csv',rows)
 print('Wrote',len(rows),'APS sensitivity rows. Primary results unchanged.')
