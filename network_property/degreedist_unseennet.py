import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
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

def degrees(G,degreef):
    d = []
    for _n in G.nodes():
        d_d = degreef(_n)
        if(d_d > 0):
            d.append(d_d)
    return np.array(d)

def degree_distro(G,degreef):
    deg= degrees(G,degreef)
    n =G.number_of_nodes()
    _pkid,_pk = np.unique(deg,return_counts=True)

    
    return _pkid,(_pk/n)


pkid_arr = []
pk_arr = []

def plot_d_dist(graphs,labels):
    print('Plotting...')
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    for g,l,c in zip(graphs,labels,colors):

        pkid,pk=degree_distro(g,g.degree)
        pkid_arr.append(pkid)
        pk_arr.append(pk)
        # s = UnivariateSpline(pkid, pk, s=5)
        # xs = np.linspace(1, 1000, 1000)
        # ys = s(xs)
        # inpkid,inpk=degree_distro(g,g.in_degree)
        # outpkid,outpk=degree_distro(g,g.out_degree)
        # plt.loglog(inpkid,inpk,'bo',color='red',label='$in \: p_k$',alpha=0.9)
        # plt.loglog(outpkid,outpk,'bo',color='blue',label='$out \: p_k$',alpha=0.7)
        plt.loglog(pkid,pk,'bo',label=l,alpha=0.5,color=c)
        #plt.loglog(xs,ys,'-')

    plt.grid()
    plt.legend(loc="best")
    plt.xlabel('$Degree(\log(k))$')
    plt.ylabel('$Probability(\log(p_k))$')
    plt.title('Degree distribution for train set of 10 networks')
    print('Done.')
    #plt.show()
def plot_from_file():
    pkid = np.loadtxt('pkid_unseen.csv')
    pk = np.loadtxt('pk_unseen.csv')
    s = UnivariateSpline(np.sort(np.log10(pkid)), np.sort(np.log10(pk))[::-1], s=5)
    xs = np.sort(np.log10(pkid))
    ys = s(xs)
    plt.plot(np.log10(pkid),np.log10(pk),'bo',label='$p_k$',alpha=0.2,color='red')
    plt.plot(xs,ys,'-',label='Univariate spline interpolation',linewidth=2,color='blue',alpha=0.7)
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel('$Degree(\log(k))$')
    plt.ylabel('$Probability(\log(p_k))$')
    plt.title('Degree distribution for unseen data set of 5 networks')
    plt.savefig('deg_dist_unseen')
    plt.show()
if __name__ == "__main__":
    #g = 
   # plot_d_dist(g)
    # graphs = []
    # labels = []
    # print('Parsing networks...')
    # print(dirs)
    # for i,d in enumerate(dirs):
    #     graphs.append(get_graph(d))
    #     labels.append(d)
    #     # if(i==2):
    #         # break
    # print('Done.')
    # plot_d_dist(graphs,labels)

    # np.savetxt('pkid_unseen.csv', np.concatenate(pkid_arr), delimiter=',', fmt='%s')
    # np.savetxt('pk_unseen.csv', np.concatenate(pk_arr), delimiter=',', fmt='%s')

    plot_from_file()