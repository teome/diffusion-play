#%%[markdown]
# # Experiments for the basics of nn and torch implementations
#%%
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
from pathlib import Path

#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
from fastcore.test import test_close

torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray'

path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'
with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin1')
x_train, y_train, x_valid, y_valid = map(torch.tensor, [x_train, y_train, x_valid, y_valid])

#%%[markdown]
# ## Experimenting with the basics for logsoftmax -- yet again

#%%
# setup some vars
n_train = 64
bs = 4
n_classes = 10
probs = torch.randn((bs, n_classes))
y = torch.randint(0, 9, (bs,))

#%%
x = probs
a = x.max(1, keepdim=True).values
x_norm = x - a
numerator = x_norm
logsumexp = x_norm.exp().sum(1, keepdim=True).log()
numerator.shape, logsumexp.shape
logsoftmax = numerator - logsumexp
logsoftmax

#%%
# Check against a version that doesn't use the normalisation with max
(torch.exp(x) / torch.exp(x).sum(1, keepdim=True)).log()

#%%
F.log_softmax(x, 1)
#%%
# wrap into function
def log_softmax(x):
    x_norm = x - x.max(1, keepdim=True).values
    numerator = x_norm
    logsumexp = x_norm.exp().sum(1, keepdim=True).log()
    logsoftmax = numerator - logsumexp
    return logsoftmax

assert torch.allclose(log_softmax(probs), F.log_softmax(probs, 1)), 'did not find equality with torch in log_softmax implementation'

#%%
log_likelihood = log_softmax(probs)
preds = log_likelihood[range(log_likelihood.shape[0]), y]
nll = -preds.mean()

print(nll, F.nll_loss(F.log_softmax(probs, 1), y), F.cross_entropy(probs, y))

#%%
# wrap into function
def cross_entropy(probs, targets):
    log_likelihood = log_softmax(probs)
    preds = log_likelihood[range(log_likelihood.shape[0]), targets]
    nll = -preds.mean()
    return nll

assert torch.allclose(cross_entropy(probs, y), F.cross_entropy(probs, y)), 'did not find equality with torch in cross_entropy implementation'

# ## TODO implement the basic model and training loop to test

#%%[markdown]
# # Datasets, batching and sampling
#%%
class Dataset:
    def __init__(self, inputs, targets):
        self.inputs, self.targets = inputs, targets 
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
#%%
train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)
assert len(train_ds) == len(x_train)
assert len(valid_ds) == len(x_valid)

#%%
xb, yb = train_ds[0:5]
assert xb.shape == (5, 28*28)
# %%
class Dataloader:
    def __init__(self, ds, bs): self.ds, self.bs = ds, bs
    def __iter__(self):
        for i in range(0, len(self.ds), self.bs): yield self.ds[i:i+bs]

#%%
train_dl = Dataloader(train_ds, bs)
valid_dl = Dataloader(valid_ds, bs)
# %%
xb, yb = next(iter(train_dl))
xb
# %%
plt.imshow(xb[0].view(28,28))
# %%
# random sampling
import random
#%%
class Sampler:
    def __init__(self, ds, shuffle=False): self.n, self.shuffle = len(ds), shuffle
    def __iter__(self):
        res = list(range(self.n))
        if self.shuffle:
            random.shuffle(res)
        return iter(res)
# %%
ss = Sampler(train_ds)
# %%
it = iter(ss)
for o in range(5): print(next(it))
# %%
for i, n in enumerate(ss):
    print( i)
    if i > 4: break

# %%
from itertools import islice
for n in islice(ss, 5): print(n)
# %%
def chunked(xs, sz, drop_last=False):
    idx = 0
    xslen = len(xs)
    while True:
        endidx = idx + sz
        print(idx, endidx, sz, xslen)
        if (endidx <= xslen and drop_last) or idx < xslen:
            startidx = idx
            idx += sz
            yield xs[startidx:min(endidx, xslen)]
        else: raise StopIteration

#%%
xs = train_ds[0:5]
xb = chunked(xs, 2, False)

#%%
next(xb)
#%%
for b in chunked(train_ds[0:5], 2, False):
    print(b)

#%%
# implementing BatchSampler to encapsulate this slicing/limiting
class BatchSampler:
    def __init__(self, sampler, ds, drop_last=False): 
        self.sampler, ds, drop_last = sampler, ds, drop_last
        self.idx = 0
    def __iter__(self):
        