"""224px pretrained-ResNet18 extension; independent from the primary run."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision.models import resnet18,ResNet18_Weights
from run import ROOT,retained_indices,context_menu
from models import ContextBatchNorm2d,configure
from conformal import calibrate,measure

torch.set_num_threads(2);torch.hub.set_dir(str(ROOT/'pretrained'))
OUT=ROOT/'results_highres';OUT.mkdir(exist_ok=True)

def load_data():
 high=np.load(ROOT/'data'/'pneumoniamnist_224.npz');low=np.load(ROOT/'data'/'pneumoniamnist.npz');keep=retained_indices(low)
 filtered={}
 for s in ['train','val','test']:
  assert np.array_equal(high[s+'_labels'],low[s+'_labels'])
  filtered[s+'_images']=high[s+'_images'][keep[s]];filtered[s+'_labels']=high[s+'_labels'][keep[s]]
 keep2=retained_indices(filtered);d={};indices={}
 for s in keep:
  indices[s]=keep[s][keep2[s]];x=high[s+'_images'][indices[s]]
  if x.ndim==3:x=x[:,None]
  d[s]=(torch.from_numpy(x.copy()),torch.from_numpy(high[s+'_labels'][indices[s]].ravel().copy()).long())
 np.savez(OUT/'retained_indices.npz',**indices)
 (OUT/'data_provenance.json').write_text(json.dumps({'archive_sha256':hashlib.sha256((ROOT/'data'/'pneumoniamnist_224.npz').read_bytes()).hexdigest(),'counts':{s:len(v[1]) for s,v in d.items()},'additional_224_byte_removals':{s:len(keep[s])-len(indices[s]) for s in keep}},indent=2))
 return d

def replace_bn(parent):
 for name,child in list(parent.named_children()):
  if isinstance(child,nn.BatchNorm2d):
   new=ContextBatchNorm2d(child.num_features,eps=child.eps,momentum=child.momentum,affine=child.affine,track_running_stats=child.track_running_stats)
   new.load_state_dict(child.state_dict());new.fused_target=True;setattr(parent,name,new)
  else:replace_bn(child)

def model(seed=0,pretrained=True):
 torch.manual_seed(seed)
 m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None);m.fc=nn.Linear(m.fc.in_features,2);replace_bn(m);return m

def prepare(x,device):
 x=x.to(device).float().expand(-1,3,-1,-1)/255
 mean=torch.tensor([.485,.456,.406],device=device).reshape(1,3,1,1)
 std=torch.tensor([.229,.224,.225],device=device).reshape(1,3,1,1)
 return (x-mean)/std

@torch.inference_mode()
def plain(m,x,device):
 configure(m,'source');return torch.cat([m(prepare(v,device)).softmax(-1).cpu() for v in x.split(32)]).numpy()

@torch.inference_mode()
def infer(m,x,pool,ix,mode,device,block=4):
 configure(m,mode,ix.shape[1]+1)
 if mode=='source':
  p=plain(m,x,device);return np.broadcast_to(p,(len(ix),*p.shape)).copy()
 result=[];b=ix.shape[1]+1
 for j,ids in enumerate(ix):
  companions=prepare(pool[ids],device);parts=[]
  for query in x.split(block):
   q=prepare(query,device);batch=torch.cat([q[:,None],companions[None].expand(len(q),-1,-1,-1,-1)],1)
   parts.append(m(batch.flatten(0,1)).reshape(len(q),b,-1)[:,0].softmax(-1).cpu())
  result.append(torch.cat(parts).numpy());print('context',mode,j+1,'/',len(ix),flush=True)
 return np.stack(result)

def run(seeds,device):
 d=load_data();ix,names=context_menu(d['train'][1].numpy(),16);order=np.random.default_rng(20406).permutation(len(d['test'][1]));cal,ev=order[:len(order)//2],order[len(order)//2:]
 (OUT/'contexts.json').write_text(json.dumps({'names':names,'indices':ix.tolist(),'batch':16},indent=2))
 for seed in seeds:
  dest=OUT/f'seed{seed}';dest.mkdir(exist_ok=True);m=model(seed).to(device);checkpoint=dest/'model.pt'
  if checkpoint.exists() and (dest/'training.json').exists():m.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
  else:
   dl=DataLoader(TensorDataset(*d['train']),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
   opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);hist=[];best=float('inf');start=time.time()
   for epoch in range(5):
    m.train();total=0
    for x,y in dl:
     opt.zero_grad();loss=nn.functional.cross_entropy(m(prepare(x,device)),y.to(device));loss.backward();opt.step();total+=float(loss.detach().cpu())*len(x)
    p=plain(m,d['val'][0],device);y=d['val'][1].numpy();nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean())
    hist.append(dict(epoch=epoch+1,train_nll=total/len(dl.dataset),val_nll=nll,val_accuracy=float((p.argmax(-1)==y).mean())))
    if nll<best:best=nll;torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},checkpoint)
    print('HIGHRES',seed,'epoch',epoch+1,'seconds',round(time.time()-start),'val nll',round(nll,4),flush=True)
   (dest/'training.json').write_text(json.dumps(hist,indent=2));m.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
  for mode in ['source','target']:
   path=dest/f'p_clean_{mode}_b16.npz'
   if path.exists():continue
   start=time.time();p=infer(m,d['test'][0],d['train'][0],ix,mode,device);y=d['test'][1].numpy();np.savez_compressed(path,p=p,y=y,cal=cal,test=ev,contexts=ix)
   rows=[dict(method=method,**measure(p[:,ev],y[ev],t),thresholds=[float(v) if np.isfinite(v) else "Infinity" for v in t]) for method,t in calibrate(p[:,cal],y[cal]).items()]
   (dest/f'metrics_{mode}.json').write_text(json.dumps(rows,indent=2));print('HIGHRES EVALUATED',seed,mode,'seconds',round(time.time()-start),'flip',round(rows[0]['class_flip_rate'],4),'nominal joint',round(rows[0]['simultaneous_coverage'],4),flush=True)
  del m
  if device=='mps':torch.mps.empty_cache()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));parser.add_argument('--device',default='mps');a=parser.parse_args();run(a.seeds,a.device)
