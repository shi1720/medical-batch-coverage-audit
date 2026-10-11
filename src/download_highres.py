import hashlib
import requests
from run import ROOT
from medmnist import INFO
p=ROOT/'data'/'pneumoniamnist_224.npz'
if not p.exists():
 with requests.get(INFO['pneumoniamnist']['url_224'],stream=True,timeout=120) as r:
  r.raise_for_status()
  with p.open('wb') as f:
   for chunk in r.iter_content(1024*1024):f.write(chunk)
assert hashlib.md5(p.read_bytes()).hexdigest()==INFO['pneumoniamnist']['MD5_224']
print('224px archive verified',p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
