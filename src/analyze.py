"""Recompute every reported number from saved per-query probabilities."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from conformal import calibrate,measure
ROOT=Path(__file__).resolve().parents[1]
METHODS=['nominal','context','bonferroni','corrected_maxrank','envelope']
DATASETS=['pneumoniamnist','breastmnist','dermamnist']
ARCHS=['cnn_bn','resnet_bn','cnn_gn']

def rows():
 out=[]
 for folder in sorted((ROOT/'results').glob('*_*_[0-4]')):
  dataset,arch,seed=folder.name.split('_',1)[0],folder.name.rsplit('_',1)[0].split('_',1)[1],int(folder.name.rsplit('_',1)[1])
  for path in sorted(folder.glob('p_*.npz')):
   _,condition,mode,batch=path.stem.split('_');z=np.load(path);p,y,cal,ev=[z[k] for k in ['p','y','cal','test']]
   assert p.shape[1]==len(y) and np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=2e-6)
   assert not set(cal)&set(ev) and len(cal)+len(ev)==len(y)
   for method,t in calibrate(p[:,cal],y[cal]).items():
    q=p[:,ev]; yp=y[ev];r=measure(q,yp,t)
    r.update(dataset=dataset,arch=arch,seed=seed,condition=condition,mode=mode,batch=int(batch[1:]),method=method,ncal=len(cal),neval=len(ev),ncontexts=len(p),accuracy=float((q[0].argmax(-1)==yp).mean()))
    try:r['auc']=float(roc_auc_score(yp,q[0,:,1])) if q.shape[-1]==2 else float(roc_auc_score(yp,q[0],multi_class='ovr'))
    except ValueError:r['auc']=None
    out.append(r)
 return out

def write_csv(path,r):
 if r:
  with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)

def sensitivity():
 out=[]
 for dataset in DATASETS:
  for arch in ARCHS[:2]:
   path=ROOT/'results'/f'{dataset}_{arch}_0'/'p_clean_target_b16.npz'
   if not path.exists():continue
   z=np.load(path);p,y=z['p'],z['y'];n=len(y)
   for seed in range(100):
    ix=np.random.default_rng(9000+seed).permutation(n);cal,ev=ix[:n//2],ix[n//2:]
    for method,t in calibrate(p[:,cal],y[cal]).items():out.append(dict(dataset=dataset,arch=arch,split=seed,method=method,**measure(p[:,ev],y[ev],t)))
 write_csv(ROOT/'results'/'split_sensitivity.csv',out)

def bootstrap():
 out=[]
 for dataset in DATASETS:
  for arch in ARCHS[:2]:
   effects=[]
   for seed in range(5):
    path=ROOT/'results'/f'{dataset}_{arch}_{seed}'/'p_clean_target_b16.npz'
    if not path.exists():break
    z=np.load(path);p,y,cal,ev=[z[k] for k in ['p','y','cal','test']];ts=calibrate(p[:,cal],y[cal]);q=p[:,ev];yp=y[ev]
    v={}
    for m,t in ts.items():
     inc=(1-q)<=t[:,None,None];truth=inc[:,np.arange(len(ev)),yp]
     v[m]=(truth.all(0).astype(float),truth.mean(0),inc.sum(-1).mean(0))
    effects.append(np.stack([v['context'][1]-v['context'][0],v['corrected_maxrank'][0]-v['context'][0],v['corrected_maxrank'][2]-v['context'][2],v['envelope'][0]-v['nominal'][0]],axis=1))
   if len(effects)!=5:continue
   effect=np.mean(effects,axis=0);rng=np.random.default_rng(410);means=effect[rng.integers(0,len(effect),size=(2000,len(effect)))].mean(1)
   for i,name in enumerate(['context_mean_minus_joint','maxrank_minus_context_coverage','maxrank_minus_context_size','envelope_minus_nominal_coverage']):
    out.append(dict(dataset=dataset,arch=arch,effect=name,mean=float(effect[:,i].mean()),lower=float(np.quantile(means[:,i],.025)),upper=float(np.quantile(means[:,i],.975)),bootstrap_unit='query; conditional on five fixed models and calibration sets'))
 write_csv(ROOT/'results'/'paired_bootstrap.csv',out)

if __name__=='__main__':
 r=rows();write_csv(ROOT/'results'/'all_metrics.csv',r);sensitivity();bootstrap()
 primary=[v for v in r if v['condition']=='clean' and v['batch']==16 and v['mode'] in ['source','target']]
 groups={}
 for a in primary:
  key=(a['dataset'],a['arch'],a['mode'],a['method']);groups.setdefault(key,[]).append(a)
 summary=[]
 for key,rs in groups.items():
  v=dict(zip(['dataset','arch','mode','method'],key));v['nseeds']=len(rs)
  for k in ['simultaneous_coverage','min_context_coverage','mean_context_coverage','mean_size','intersection_size','class_flip_rate','probability_range','accuracy','auc']:
   vals=np.array([a[k] for a in rs if a[k] is not None]);v[k+'_mean']=float(vals.mean()) if len(vals) else None;v[k+'_sd']=float(vals.std(ddof=1)) if len(vals)>1 else None
  summary.append(v)
 (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2));write_csv(ROOT/'results'/'summary.csv',summary)
 print('Probability files analyzed:',len(r)//5,'of 75 primary expected, plus secondary files. Primary complete groups:',sum(v['nseeds']==5 for v in summary))
