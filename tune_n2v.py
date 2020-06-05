#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gen_n2v import getNode2Vec
import itertools
import multiprocessing as mp
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


n_walks = [10, 50, 100]
l_walk = [5, 20, 60]
dim = [8, 16, 64]
q = [0.1, 0.4, 0.8]
comb = [q, n_walks, l_walk, dim]
comb = list(itertools.product(*comb))


# In[3]:


def calc(option, i):
    getter = getNode2Vec(path='./data/Caltech36/Caltech36',
                         q=option[0],
                         num_walks=option[1],
                         walk_length=option[2],
                         dim=option[3])
    getter.get_n2v(_id=i)


# In[4]:


proc = []
for i, option in enumerate(comb):
    calc(option,i)

#     p = mp.Process(target=calc, args=(option, i))
#     p.start()
#     proc.append(p)
# for p in proc:
#     p.join()


# In[ ]:




