# %%
from pathlib import Path
import pickle, gzip, math, os, time, shutil, matplotlib as mpl, matplotlib.pyplot as plt

# %%
MNIST_URL = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
path_data = Path('./data')
path_data.mkdir(exist_ok=True)
path_gz = path_data / 'mnist.pkl.gz'

# %%
import urllib.request

if not path_gz.exists():
    urllib.request.urlretrieve(MNIST_URL, path_gz)


# %%
with gzip.open(path_gz, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin1')

lst1 = list(x_train[0])

#%%[markdowm]
# ## Playing with iterators and itertools for arrays

#%%
def chunks(x, sz):
    for i in range(0, len(x), sz): yield x[i:i+sz]

#%%
from itertools import islice
it = iter(lst1)

img = list(iter(lambda: list(islice(it, 28)), []))

# img = list(iter(list))

# %%
plt.imshow(img)

#%%[markdown]
# ## Manual Matrix class

#%%
class Matrix:
    def __init__(self, xs): self.xs = xs
    def __getitem__(self, idxs): return self.xs[idxs[0]][idxs[1]]

#%%
# Testing the Matrix class
m = Matrix([[1, 2, 3], [4, 5, 6]])
print(m[0, 0])  # Output: 1 
print(m[1, 2])  # Output: 6
print(m[1, 1])  # Output: 5

#%%
# Now move to pytorch
import torch
from torch import tensor

#%%
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
#%%
print(y_train.min(), y_train.max())
print(y_train.max(), y_train.shape)

#%%
plt.imshow(x_train[0].reshape(28, 28))

#%%[markdown]
# ## Random numbers

# Based on Wichmann Hill algorithm in Python<2.3

#%%
rnd_state = None
def seed(a):
    global rnd_state
    a, x = divmod(a, 30268)
    a, y = divmod(a, 30306)
    a, z = divmod(a, 30322)
    rnd_state = int(x)+1, int(y)+1, int(z)+1

#%%
seed(457428938475)
rnd_state

#%%
def rand():
    global rnd_state
    x, y, z = rnd_state
    x = (171 * x) % 30269
    y = (172 * x) % 30307
    z = (170 * x) % 30323
    rnd_state = x, y, z
    return (x / 30269 + y / 30307 + z / 30323) % 1.0

#%%
rand(), rand(), rand()

#%%
# Demo of random generation gotcha. Generator is just copied with state, not reinit
if os.fork(): print(f'In parent: {rand()}')
else:
    print(f'In child: {rand()}')
    os._exit(os.EX_OK)

#%%
# And in torch - it doesn't set generator differently
if os.fork(): print(f'In parent: {torch.rand(1)}')
else:
    print(f'In child: {torch.rand(1)}')
    os._exit(os.EX_OK)

#%%
# And in python itself it does remember to reinitialise the stream in each fork
from random import random
if os.fork(): print(f'In parent: {random()}')
else:
    print(f'In child: {random()}')
    os._exit(os.EX_OK)

#%%
plt.plot([rand() for _ in range(50)]);
#%%
plt.hist([rand() for _ in range(10000)]);

#%%
list(chunks([rand() for _ in range(10)], 2))

#%%
%timeit -n 10 list(chunks([rand() for _ in range(7840)], 10))
#%%
%timeit -n 10 torch.randn(784, 10)
