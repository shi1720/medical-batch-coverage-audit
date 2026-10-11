"""Check release completeness and integrity without downloading source images."""
import hashlib,json,re,shutil,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
primary=list((ROOT/'results').glob('*_*_[0-4]/training.json'))
high=list((ROOT/'results_highres').glob('seed[0-4]/training.json'))
assert len(primary)==45 and len(high)==5
for path in primary+high:
 assert len(json.loads(path.read_text()))==(5 if 'results_highres' in str(path) else 25)
 assert (path.parent/'model.pt').is_file()
files=list((ROOT/'results').glob('*_*_[0-4]/p_*.npz'))+list((ROOT/'results_highres').glob('seed[0-4]/p_*.npz'))
assert len(files)==133,(len(files),'expected 75 primary, 48 diagnostic, 10 higher-resolution')
for path in files:
 with np.load(path) as z:
  p,y,cal,test=[z[k] for k in ['p','y','cal','test']]
  assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
  assert np.allclose(p.sum(-1),1,atol=2e-6)
  assert p.shape[1]==len(y) and np.array_equal(np.sort(np.r_[cal,test]),np.arange(len(y)))
  assert not set(cal)&set(test)
  assert z['contexts'].shape[0]==p.shape[0]
for folder,count in [('results',75),('results_highres',10)]:
 summary=json.loads((ROOT/folder/'summary.json').read_text())
 assert len(summary)==count and all(r['nseeds']==5 for r in summary)
 replay=json.loads((ROOT/folder/'replay_verification.json').read_text())
 assert len(replay)==count
 for row in replay:
  name=(f"{row['dataset']}_{row['arch']}_{row['seed']}" if folder=='results' else f"seed{row['seed']}")
  dest=ROOT/folder/name
  assert hashlib.sha256((dest/'model.pt').read_bytes()).hexdigest()==row['checkpoint_sha256']
  assert hashlib.sha256((dest/f"p_clean_{row['mode']}_b16.npz").read_bytes()).hexdigest()==row['probability_sha256']
def reject_constant(value):
 raise ValueError('Nonstandard JSON constant: '+value)
for folder in ['results','results_highres']:
 for path in (ROOT/folder).rglob('*.json'):
  json.loads(path.read_text(),parse_constant=reject_constant)
tex='\n'.join(p.read_text() for p in (ROOT/'paper').glob('*.tex'))
assert not re.search(r'\bpending\b|\bis running\b|\bare running\b|\bnan\b',tex,re.I)
bib=(ROOT/'paper'/'references.bib').read_text();keys=set(re.findall(r'@\w+\{([^,]+),',bib))
cited=set(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(','))
assert cited<=keys,(cited-keys)
pdf_pages=None
if shutil.which('pdfinfo'):
 info=subprocess.check_output(['pdfinfo',str(ROOT/'paper'/'main.pdf')],text=True)
 pdf_pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pdf_pages==4
result={'pdf_pages':pdf_pages,'replayed_predictions':85,'primary_models':len(primary),'higher_resolution_models':len(high),'saved_probability_files':len(files),'references':len(keys),'all_citations_resolved':True,'source_images_required_for_this_check':False}
(ROOT/'release_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
