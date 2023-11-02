#%% [markdown]
# # Denoising Diffusion Implicit Models -- DDIM
# %%
import pickle,gzip,math,os,time,shutil,torch,random,logging
import fastcore.all as fc,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
from collections.abc import Mapping
from pathlib import Path
from functools import partial

from fastcore.foundation import L
import torchvision.transforms.functional as TF,torch.nn.functional as F
from torch import tensor,nn,optim
from torch.utils.data import DataLoader,default_collate
from torch.nn import init
from torch.optim import lr_scheduler

from miniai.datasets import *
from miniai.conv import *
from miniai.learner import *
from miniai.activations import *
from miniai.init import *
from miniai.sgd import *
from miniai.resnet import *
from miniai.augment import *
from miniai.accel import *
from miniai.fid import *

# %%
from fastprogress.fastprogress import progress_bar
# %%
from torcheval.metrics import MulticlassAccuracy
from datasets import load_dataset,load_dataset_builder

mpl.rcParams['image.cmap'] = 'gray_r'
logging.disable(logging.WARNING)

# %%
xl,yl = 'image','label'
name = "fashion_mnist"
dsd = load_dataset(name)

# %%
from diffusers import UNet2DModel, DDIMPipeline, DDPMPipeline, DDIMScheduler, DDPMScheduler

# %%[markdown]
# ## Diffusers DDPM Scheduler

# %%
# We have our model trained in miniai that we wrapped in UNet and overrode the forward method to return the `.sample()`.
# Using diffusers here we can't do that, it expects it as it should be in the original class. So, we can just create
# class to wrap it to do nothing, so that we can still torch.load without complaints
class UNet(UNet2DModel): pass

# %%
model = torch.load('../models/fashion_ddpm3_25.pkl', map_location=torch.device(def_device))

# %%
sched = DDPMScheduler(beta_end=0.01)

# %%
x_t = torch.randn((32, 1, 32, 32)).to(def_device)
# %%
t = 999
t_batch = torch.full((len(x_t),), t, device=x_t.device, dtype=torch.long)
with torch.no_grad(): noise = model(x_t, t_batch).sample

# %%
res = sched.step(noise, t, x_t)
res.prev_sample.shape

# %%
%%time
sz = (2048, 1, 32, 32)
x_t = torch.randn(sz).to(def_device)
preds = []

for t in progress_bar(sched.timesteps):
    with torch.no_grad(): noise = model(x_t, t).sample
    x_t = sched.step(noise, t, x_t).prev_sample
    preds.append(x_t.float().cpu())
# %%
s = preds[-1].clamp(-0.5,0.5)*2

# %%
show_images(s[:25], imsize=1.5)

# %%
# cmodel = torch.load('models/data_aug2.pkl')
cmodel = torch.load('/content/data_aug2.pkl', map_location=def_device)
del(cmodel[8])
del(cmodel[7])

# %%
@inplace
def transformi(b): b[xl] = [F.pad(TF.to_tensor(o), (2,2,2,2))*2-1 for o in b[xl]]

bs = 2048
tds = dsd.with_transform(transformi)
dls = DataLoaders.from_dd(tds, bs, num_workers=fc.defaults.cpus)

dt = dls.train
xb,yb = next(iter(dt))

ie = ImageEval(cmodel, dls, cbs=[DeviceCB()])

# %%
ie.fid(s),ie.kid(s)

# %%
ie.fid(xb),ie.kid(xb)

# %% [markdown]
# ## Diffusers DDIM Scheduler

# %%
sched = DDIMScheduler(beta_end=0.01)
sched.set_timesteps(333)

# %%
def diff_sample(model, sz, sched, **kwargs):
    x_t = torch.randn(sz).to(def_device)
    preds = []
    for t in progress_bar(sched.timesteps):
        with torch.no_grad(): noise = model(x_t, t).sample
        x_t = sched.step(noise, t, x_t, **kwargs).prev_sample
        preds.append(x_t.float().cpu())
    return preds

# %%
# eta param determines how much of the noise that the schedule says should be added is added
preds = diff_sample(model, sz, sched, eta=1.)
s = (preds[-1] * 2).clamp(-1, 1)

# %%
show_images(s[:25], imsize=1.5)

# %%
ie.fid(s),ie.kid(s)

# %%
sched.set_timesteps(200)
preds = diff_sample(model, sz, sched, eta=1.)
s = (preds[-1]*2).clamp(-1,1)
print(ie.fid(s),ie.kid(s))
show_images(s[:25], imsize=1.5)

# %%
sched.set_timesteps(100)
preds = diff_sample(model, sz, sched, eta=1.)
s = (preds[-1]*2).clamp(-1,1)
print(ie.fid(s),ie.kid(s))
show_images(s[:25], imsize=1.5)

# %%
sched.set_timesteps(50)
preds = diff_sample(model, sz, sched, eta=1.)
s = (preds[-1]*2).clamp(-1,1)
print(ie.fid(s),ie.kid(s))
show_images(s[:25], imsize=1.5)

# %%
sched.set_timesteps(25)
preds = diff_sample(model, sz, sched, eta=1.)
s = (preds[-1]*2).clamp(-1,1)
print(ie.fid(s),ie.kid(s))
show_images(s[:25], imsize=1.5)


