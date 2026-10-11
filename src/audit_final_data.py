"""Final data audit including published lesion-disjoint dermatology correction."""
import csv,hashlib,json
import numpy as np
from run import ROOT,retained_indices,data
out=[]
for name in ['pneumoniamnist','breastmnist','dermamnist']:
 filename='dermamnist_corrected_28.npz' if name=='dermamnist' else name+'.npz'
 path=ROOT/'data'/filename;z=np.load(path);keep=retained_indices(z)
 row=dict(dataset=name,archive=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw={},retained={s:len(v[1]) for s,v in data(name).items()})
 sets={}
 for s in ['train','val','test']:
  hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in z[s+'_images']];sets[s]=set(hashes)
  row['raw'][s]={'n':len(hashes),'unique_byte_hashes':len(sets[s]),'label_counts':np.bincount(z[s+'_labels'].ravel()).tolist()}
 row['raw_cross_split_byte_overlap']={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
 if name=='dermamnist':
  r=list(csv.DictReader((ROOT/'data'/'derma_corrected.csv').open()));lesions={s:{v['lesion_id'] for v in r if v['split']==s} for s in sets}
  row['lesion_overlap']={a+'_'+b:len(lesions[a]&lesions[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
  row['metadata_sha256']=hashlib.sha256((ROOT/'data'/'derma_corrected.csv').read_bytes()).hexdigest()
  testmeta=[v for v in r if v['split']=='test'];seen=set();ix=[]
  for i in keep['test']:
   lesion=testmeta[i]['lesion_id']
   if lesion not in seen:ix.append(i);seen.add(lesion)
  keep['test']=np.asarray(ix)
 np.savez(ROOT/'results'/f'{name}_final_retained_indices.npz',**keep);out.append(row)
(ROOT/'results'/'final_data_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
