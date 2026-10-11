"""Only measure training throughput/memory; no held-out performance selection."""
import time,json,torch
from torchvision.models import resnet18,ResNet18_Weights
from run import ROOT
# Keep weight downloads inside this project, not the user's home cache.
torch.hub.set_dir(str(ROOT/'pretrained'))
torch.set_num_threads(2);torch.manual_seed(0)
m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);m.fc=torch.nn.Linear(m.fc.in_features,2);m.to('mps');m.train()
x=torch.randn(16,3,224,224,device='mps');y=torch.arange(16,device='mps')%2
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
start=time.time()
for i in range(5):
 opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward();opt.step();torch.mps.synchronize()
 print('step',i,'elapsed',round(time.time()-start,2),'allocated',torch.mps.current_allocated_memory(),flush=True)
r={'five_steps_seconds':time.time()-start,'batch':16,'allocated_bytes':torch.mps.current_allocated_memory(),'driver_allocated_bytes':torch.mps.driver_allocated_memory()}
(ROOT/'highres_feasibility.json').write_text(json.dumps(r,indent=2))
