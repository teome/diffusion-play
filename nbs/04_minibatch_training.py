#%%[markdown]
# # Experiments for the basics of nn and torch implementations
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
