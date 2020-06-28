import networkx as nx
import numpy as np 
import os 


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_path = '' + os.path.abspath(os.path.join(THIS_FOLDER, os.pardir)) + '/unseen-data/'
dirs = [
    _dir for _dir in os.listdir(f'{data_path}')
    if os.path.isdir(f"{data_path}{_dir}")
]

def get_graph(_dir):
    p = f'{data_path}{_dir}/{_dir}.net'
    return nx.Graph(nx.read_pajek(p), nodetype=int)

avg_c = []
if __name__ == "__main__":
    for d in dirs:
        print('Calculating clustering for ', d,'...')
        res = {}
        res = nx.clustering(get_graph(d))
        val = np.mean(list(res.values()))
        print(val)
        avg_c.append({d:val})
