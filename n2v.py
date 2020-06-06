#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


import networkx as nx
import pandas as pd
import numpy as np
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import os
import multiprocessing as mp
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


# In[2]:


class getNode2Vec:
    def __init__(self, path, dim, num_walks, q, walk_length):
        self.G = nx.Graph(nx.read_pajek(path + '.net'), nodetype=int)
        raw_data = pd.read_csv(path + '.raw_data', sep='\t')
        self.pos_e = raw_data[raw_data['label'] == 1][['from_id',
                                                       'to_id']].values
        self.e = raw_data[['from_id', 'to_id']].values
        self.nodes = raw_data[['from_id', 'to_id']]
        self.labels = raw_data[['label']]
        self.dim = dim
        self.path = path
        self.num_walks = num_walks
        self.q = q
        self.walk_length = walk_length

    def remove_edges(self):
        for val in self.pos_e:
            self.G.remove_edge(str(val[0]), str(val[1]))

    def gen_node2vec(self):
        # Remove positive edges from graph:
        self.remove_edges()
        node2vec = Node2Vec(
            self.G,
            dimensions=self.dim,  #lower dim
            walk_length=self.walk_length,  #shorter walk than 100
            num_walks=self.num_walks,  # bigger number than 10
            workers=5,
            p=1,  #p = 1
            q=self.q,  #q < 1
            temp_folder='./temp/')  # Use temp_folder for big graphs
        # Embed nodes
        n2v_df = pd.DataFrame()
        model = node2vec.fit()  #Use over gensim word2vec
        edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
        # Edges:
        for e in self.e:
            v = edges_embs[(str(e[0]), str(e[1]))]
            res = dict(('d' + str(i), el) for i, el in enumerate(v))
            n2v_df = n2v_df.append(res, ignore_index=True)

        return pd.concat([self.nodes, n2v_df, self.labels], axis=1)

    def get_n2v(self):
        data = gen_node2vec()
        data.to_csv(self.path + 'emb.csv')


# In[3]:


def main(rootFolder):
    #get all dirs in data folder and parse every network there
    dirs = [
        _dir for _dir in os.listdir(f'./{rootFolder}/')
        if os.path.isdir(f"./{rootFolder}/{_dir}")
    ] 
    for x in dirs:
        print(x)

    def ww(p):
        #Configuration id=68
        m = getNode2Vec(p, dim=64, num_walks=50, q=0.8, walk_length=20)
        emb_model = m.gen_node2vec()
        emb_model.to_csv(p + 'emb.csv.gz', compression='gzip', index=False)
        print('FINISHED WITH:', p)

    proc = []
    for net in dirs:
        path = f"./{rootFolder}/{net}/{net}"
        p = mp.Process(target=ww, args=(path, ))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


# In[4]:


def merge_data(rootFolder):
    dirs = [
        _dir for _dir in os.listdir(f'./{rootFolder}/')
        if os.path.isdir(f"./{rootFolder}/{_dir}")
    ]
    for x in dirs:
        print(x)
    networks_data = []
    for net in dirs:
        path = f"./{rootFolder}/{net}/{net}"
        data = pd.read_csv(path + 'emb.csv.gz', compression='gzip', sep=',')
        raw_data = pd.read_csv(path + '.raw_data', sep='\t')
        raw_data = raw_data[[
            "is_dorm", "is_year", "year_diff", "from_high_school",
            "to_high_school", "from_major", "to_major", "is_faculty",
            "is_gender"
        ]]
        data = pd.concat([raw_data,data],axis = 1)
        networks_data.append(data)

    networks_data = pd.concat(networks_data, axis=0)
    
    networks_data = networks_data.drop(columns=['from_id', 'to_id'])
    '''
    FOR UNSEEN DATA :
    data = pd.concat([train,test],axis=0)
    data.to_csv('./unseen-data/node2vec_v2_data.csv.gz',compression='gzip',sep='\t',index=False)
    '''
    train, test = train_test_split(networks_data, test_size=0.25)
    train.to_csv(f'./{rootFolder}/node2vec_v2_train.csv.gz', compression='gzip', sep='\t', index=False)
    networks_data.to_csv(f'./{rootFolder}/node2vec_v2_data.csv.gz', compression='gzip', sep='\t', index=False)
    print('Train:',train.info())
    print('Test:',test.info())


# In[5]:


if __name__ == "__main__":
    #main()
    print('main')
    #merge_data()


# In[9]:





# In[11]:





# In[ ]:




