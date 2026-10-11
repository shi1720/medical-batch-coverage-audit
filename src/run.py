"""Train models and save counterfactual companion-context probabilities."""
import argparse,copy,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from medmnist import INFO
from models import make_model,configure
from conformal import calibrate,measure

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

def tensor(a):
    if a.ndim==3: a=a[...,None]
    return torch.from_numpy(a.transpose(0,3,1,2).copy()).float()/127.5-1

def retained_indices(z):
    # Label-blind exact-byte de-duplication, with train > val > test priority.
    seen=set(); retained={}
    for split in ['train','val','test']:
        keep=[]
        for i,image in enumerate(z[f'{split}_images']):
            digest=hashlib.sha256(image.tobytes()).digest()
            if digest not in seen:
                keep.append(i);seen.add(digest)
        retained[split]=np.array(keep,dtype=int)
    return retained

def data(name):
    filename='dermamnist_corrected_28.npz' if name=='dermamnist' else f'{name}.npz'
    z=np.load(ROOT/'data'/filename); keep=retained_indices(z)
    if name=='dermamnist':
        rows=list(csv.DictReader((ROOT/'data'/'derma_corrected.csv').open()))
        # Published lesion-disjoint partitions; one deterministic test image per lesion.
        meta={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
        sets={s:{r['lesion_id'] for r in meta[s]} for s in meta}
        assert not any(sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')])
        mapping={v:i for i,v in enumerate(['akiec','bcc','bkl','df','mel','nv','vasc'])}
        for s in meta:
            assert len(meta[s])==len(z[f'{s}_labels'])
            assert all(mapping[r['dx']]==y for r,y in zip(meta[s],z[f'{s}_labels'].ravel()))
        seen=set(); single=[]
        for i in keep['test']:
            lesion=meta['test'][i]['lesion_id']
            if lesion not in seen:single.append(i);seen.add(lesion)
        keep['test']=np.asarray(single,dtype=int)
    return {s:(tensor(z[f'{s}_images'][keep[s]]),torch.tensor(z[f'{s}_labels'][keep[s]].ravel(),dtype=torch.long)) for s in ['train','val','test']}

@torch.inference_mode()
def plain(model,x,device='cpu'):
    configure(model,'source')
    return torch.cat([model(t.to(device)).softmax(-1).cpu() for t in x.split(128)]).numpy()

def context_menu(y,batch=16,replicates=3,seed=801):
    y=np.asarray(y); classes=np.unique(y); result=[]; names=[]
    for r in range(replicates):
        rng=np.random.default_rng(seed+r)
        result.append(rng.choice(len(y),batch-1,replace=False)); names.append(f'natural_{r}')
        ids=[]
        for i in range(batch-1):
            pool=np.flatnonzero(y==classes[i%len(classes)])
            # Each class-specific draw is independent; duplicate probability recorded in metadata.
            ids.append(rng.choice(pool))
        result.append(np.array(ids)); names.append(f'balanced_{r}')
        for c in classes:
            pool=np.flatnonzero(y==c)
            result.append(rng.choice(pool,batch-1,replace=len(pool)<batch-1))
            names.append(f'class{c}_{r}')
    return np.array(result),names


def train(name,arch,seed,epochs,device):
    dest=ROOT/'results'/f'{name}_{arch}_{seed}'
    dest.mkdir(exist_ok=True)
    d=data(name); c=INFO[name]['n_channels']; k=len(INFO[name]['label'])
    torch.manual_seed(seed); np.random.seed(seed)
    model=make_model(arch,c,k).to(device)
    checkpoint=dest/'model.pt'
    if checkpoint.exists() and (dest/'training.json').exists():
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); return model,d,dest
    dl=DataLoader(TensorDataset(*d['train']),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loss_fn=nn.CrossEntropyLoss(); best=float('inf'); history=[]; start=time.time()
    for epoch in range(epochs):
        model.train(); loss_total=0
        for x,y in dl:
            x,y=x.to(device),y.to(device);opt.zero_grad();loss=loss_fn(model(x),y);loss.backward();opt.step();loss_total+=float(loss.detach())*len(x)
        p=plain(model,d['val'][0],device);y=d['val'][1].numpy()
        nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean())
        history.append({'epoch':epoch+1,'train_nll':loss_total/len(dl.dataset),'val_nll':nll,'val_accuracy':float((p.argmax(1)==y).mean())})
        if nll < best:
            best=nll;torch.save(model.state_dict(),checkpoint)
        if epoch%5==4: print(name,arch,seed,'epoch',epoch+1,'val nll',round(nll,3),'seconds',round(time.time()-start),flush=True)
    (dest/'training.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    return model,d,dest

@torch.inference_mode()
def contexts(model,queries,pool,indices,mode,device='cpu',query_block=16):
    b=indices.shape[1]+1
    configure(model,mode,b)
    if mode=='source' or not any(getattr(m,'strength',0)>0 for m in model.modules()):
        p=plain(model,queries,device)
        return np.broadcast_to(p,(len(indices),*p.shape)).copy()
    result=[]
    for ix in indices:
        companion=pool[ix].to(device);blocks=[]
        for q in queries.split(query_block):
            q=q.to(device)
            grouped=torch.cat([q[:,None],companion[None].expand(len(q),-1,-1,-1,-1)],dim=1)
            logits=model(grouped.flatten(0,1)).reshape(len(q),b,-1)[:,0]
            blocks.append(logits.softmax(-1).cpu())
        result.append(torch.cat(blocks).numpy())
    return np.stack(result)


def run(args):
    for name in args.datasets:
        for arch in args.archs:
            for seed in args.seeds:
                model,d,dest=train(name,arch,seed,args.epochs,args.device)
                x,y=d['test']; order=np.random.default_rng(20406).permutation(len(y)); cal=order[:len(y)//2];test=order[len(y)//2:]
                ix,names=context_menu(d['train'][1].numpy(),args.batch)
                np.savez(dest/f'indices_b{args.batch}.npz',cal=cal,test=test,contexts=ix)
                (dest/f'contexts_b{args.batch}.json').write_text(json.dumps({'names':names,'batch':args.batch,'unique_counts':[len(np.unique(z)) for z in ix]},indent=2))
                modes=['source'] if arch=='cnn_gn' else args.modes
                for condition in args.conditions:
                    transform=lambda t: t if condition=='clean' else (((t+1)/2*.75+.1).clamp(0,1)*2-1)
                    for mode in modes:
                        path=dest/f'p_{condition}_{mode}_b{args.batch}.npz'
                        if path.exists():continue
                        start=time.time();p=contexts(model.to(args.inference_device),transform(x),transform(d['train'][0]),ix,mode,args.inference_device);model.to(args.device)
                        np.savez_compressed(path,p=p,y=y.numpy(),cal=cal,test=test,contexts=ix)
                        rows=[]
                        for method,t in calibrate(p[:,cal],y.numpy()[cal]).items():
                            rows.append({'method':method,**measure(p[:,test],y.numpy()[test],t),'thresholds':[float(v) if np.isfinite(v) else 'Infinity' for v in t]})
                        (dest/f'metrics_{condition}_{mode}_b{args.batch}.json').write_text(json.dumps(rows,indent=2))
                        print('EVALUATED',name,arch,seed,condition,mode,'seconds',round(time.time()-start),'flip',round(rows[0]['class_flip_rate'],3),'nominal simultaneous',round(rows[0]['simultaneous_coverage'],3),'envelope',round(rows[-1]['simultaneous_coverage'],3),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--datasets',nargs='+',default=['pneumoniamnist','breastmnist','dermamnist']);a.add_argument('--archs',nargs='+',default=['cnn_bn','resnet_bn','cnn_gn']);a.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));a.add_argument('--epochs',type=int,default=25);a.add_argument('--modes',nargs='+',default=['source','target','mixed','first']);a.add_argument('--conditions',nargs='+',default=['clean','affine']);a.add_argument('--batch',type=int,default=16);a.add_argument('--device',default='cpu');a.add_argument('--inference-device',default='mps');run(a.parse_args())
