"""Download official archives to the project, with upstream MD5 verification."""
import hashlib
from pathlib import Path
import requests
from medmnist import INFO
ROOT=Path(__file__).resolve().parents[1]
for name in ['pneumoniamnist','breastmnist']:
    p=ROOT/'data'/f'{name}.npz';p.parent.mkdir(exist_ok=True)
    if not p.exists():
        r=requests.get(INFO[name]['url'],timeout=120);r.raise_for_status();p.write_bytes(r.content)
    if hashlib.md5(p.read_bytes()).hexdigest()!=INFO[name]['MD5']:
        raise RuntimeError(f'{name}: archive checksum mismatch')
    print(name,'verified',hashlib.sha256(p.read_bytes()).hexdigest())
# Published lesion-disjoint DermaMNIST-C; upstream commit pinned for reproducibility.
base='https://raw.githubusercontent.com/kakumarabhishek/Corrected-Skin-Image-Datasets/ac655ac5a5b93264c57d69d6d520778d915bfc89/DermaMNIST/DermaMNIST_Analysis/'
for filename,remote,sha in [
 ('dermamnist_corrected_28.npz','NPZ_files/dermamnist_corrected_28.npz','c3d9cbd9008a0dcb76943d5cf38681bdc5166effc79f9e215657b859b9226aa8'),
 ('derma_corrected.csv','CSV_files/combined_metadata_corrected-HAM10000_corrected.csv','c27194d338986463966e1694d5b49fda933222368f04b316c702ac2e8a0d4f5c')]:
 p=ROOT/'data'/filename
 if not p.exists():
  r=requests.get(base+remote,timeout=120);r.raise_for_status();p.write_bytes(r.content)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise RuntimeError(f'{filename}: SHA256 mismatch')
 print(filename,'verified',sha)
