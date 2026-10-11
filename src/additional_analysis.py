"""Exploratory rare-class and held-out-menu diagnostics, without extra model fitting."""
import csv
import numpy as np
from analyze import ROOT,DATASETS,ARCHS,write_csv
from conformal import calibrate,measure
class_rows=[];menu_rows=[]
for dataset in DATASETS:
 for arch in ARCHS[:2]:
  for seed in range(5):
   path=ROOT/'results'/f'{dataset}_{arch}_{seed}'/'p_clean_target_b16.npz'
   if not path.exists():continue
   z=np.load(path);p,y,cal,ev=[z[k] for k in ['p','y','cal','test']]
   for m,t in calibrate(p[:,cal],y[cal]).items():
    for c in range(p.shape[-1]):
     selected=ev[y[ev]==c]
     if len(selected):class_rows.append(dict(dataset=dataset,arch=arch,seed=seed,method=m,label=c,n=len(selected),**measure(p[:,selected],y[selected],t)))
   # Three ordered menu replicates, each natural, balanced, then K homogeneous contexts.
   per_rep=p.shape[0]//3;seen=p[:2*per_rep];unseen=p[2*per_rep:]
   ts=calibrate(seen[:,cal],y[cal])
   for m in ['nominal','envelope']:
    q=np.full(len(unseen),ts[m][0]);menu_rows.append(dict(dataset=dataset,arch=arch,seed=seed,method=m,seen_contexts=len(seen),unseen_contexts=len(unseen),**measure(unseen[:,ev],y[ev],q)))
write_csv(ROOT/'results'/'class_metrics.csv',class_rows)
write_csv(ROOT/'results'/'held_out_menu.csv',menu_rows)
print('Wrote',len(class_rows),'class diagnostic rows and',len(menu_rows),'held-out-menu rows. No unseen-menu guarantee claimed.')
