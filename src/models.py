"""Classifiers and exact independent-context BatchNorm inference."""
import torch
from torch import nn


class ContextBatchNorm2d(nn.BatchNorm2d):
    """Training is stock BN. At inference, groups never share statistics.

    strength=0 uses frozen source statistics; strength=1 uses each group's
    population moments. Intermediate strengths mix first and second moments,
    including the between-domain mean term in the resulting variance.
    """
    strength = 0.0
    group_size = 1
    fused_target = False

    def forward(self, x):
        if self.training or self.strength == 0:
            return super().forward(x)
        b = self.group_size
        if len(x) % b:
            raise ValueError('Flattened batch must be divisible by context size')
        if self.strength == 1 and self.fused_target:
            # Fold independent query groups into channels for native fused BN.
            # Each channel still reduces only its own B,H,W companion group.
            q=len(x)//b;c=x.shape[1]
            folded=x.reshape(q,b,*x.shape[1:]).transpose(0,1).reshape(b,q*c,*x.shape[2:])
            out=torch.nn.functional.batch_norm(folded,None,None,self.weight.repeat(q),self.bias.repeat(q),True,0.,self.eps)
            return out.reshape(b,q,*x.shape[1:]).transpose(0,1).reshape_as(x)
        z = x.reshape(-1, b, *x.shape[1:])
        mean = z.mean(dim=(1, 3, 4), keepdim=True)
        var = z.var(dim=(1, 3, 4), unbiased=False, keepdim=True)
        shape = (1, 1, -1, 1, 1)
        source_mean = self.running_mean.reshape(shape)
        source_var = self.running_var.reshape(shape)
        a = self.strength
        mixed_mean = a * mean + (1-a) * source_mean
        mixed_var = a * var + (1-a) * source_var + a*(1-a)*(mean-source_mean).square()
        out = (z-mixed_mean) / (mixed_var+self.eps).sqrt()
        out = out*self.weight.reshape(shape) + self.bias.reshape(shape)
        return out.reshape_as(x)


def norm_layer(channels, kind):
    return ContextBatchNorm2d(channels) if kind == 'bn' else nn.GroupNorm(4, channels)


class ConvNet(nn.Module):
    def __init__(self, channels, classes, norm='bn'):
        super().__init__()
        blocks = []
        for width in (16, 32, 64):
            blocks += [nn.Conv2d(channels, width, 3, padding=1, bias=False),
                       norm_layer(width, norm), nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(64*3*3, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            ContextBatchNorm2d(cout), nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            ContextBatchNorm2d(cout))
        self.skip = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False), ContextBatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.body(x)+self.skip(x))


class SmallResNet(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels,16,3,padding=1,bias=False),
            ContextBatchNorm2d(16),nn.ReLU(),ResidualBlock(16,16),ResidualBlock(16,32,2),
            ResidualBlock(32,64,2),nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, classes)

    def forward(self,x):
        return self.head(self.features(x).flatten(1))


def make_model(arch, channels, classes):
    if arch == 'cnn_bn': return ConvNet(channels, classes, 'bn')
    if arch == 'cnn_gn': return ConvNet(channels, classes, 'gn')
    if arch == 'resnet_bn': return SmallResNet(channels, classes)
    raise ValueError(arch)


def configure(model, mode, group_size=1):
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, ContextBatchNorm2d)]
    for i,m in enumerate(layers):
        m.group_size=group_size
        m.strength={'source':0.,'target':1.,'mixed':.5,'first':float(i==0)}[mode]
    return model
