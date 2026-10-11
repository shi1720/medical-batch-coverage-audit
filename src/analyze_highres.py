"""Analyze the predefined higher-resolution extension from archived probabilities."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from run_highres import OUT
from conformal import calibrate,measure
from analyze import write_csv
rows=[]
for seed in range(5):
 for mode in ['source','target']:
  path=OUT/f'seed{seed}'/f'p_clean_{mode}_b16.npz'
  if not path.exists():continue
  z=np.load(path);p,y,cal,ev=[z[k] for k in ['p','y','cal','test']]
  assert np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=2e-6)
  for method,t in calibrate(p[:,cal],y[cal]).items():rows.append(dict(seed=seed,mode=mode,method=method,**measure(p[:,ev],y[ev],t),accuracy=float((p[0,ev].argmax(-1)==y[ev]).mean()),auc=float(roc_auc_score(y[ev],p[0,ev,1]))))
write_csv(OUT/'all_metrics.csv',rows)
summary=[]
for mode in ['source','target']:
 for method in ['nominal','context','bonferroni','corrected_maxrank','envelope']:
  rs=[r for r in rows if r['mode']==mode and r['method']==method]
  if not rs:continue
  v=dict(mode=mode,method=method,nseeds=len(rs))
  for k in ['simultaneous_coverage','min_context_coverage','mean_context_coverage','mean_size','class_flip_rate','accuracy','auc']:
   vals=[r[k] for r in rs];v[k+'_mean']=float(np.mean(vals));v[k+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
  summary.append(v)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));write_csv(OUT/'summary.csv',summary)
print('Higher-resolution probability files analyzed:',len(rows)//5,'of 10 expected.')
