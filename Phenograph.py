import phenograph
import pandas as pd
import numpy as np
import community
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from random import shuffle
import pickle
import time

#импорт данных
data = pd.read_csv('../../PCA_res.csv')

#ткани для первого графика
datalabels = pd.DataFrame(data['cell_type'])
labels = data['cell_type']
data = data.drop(axis='columns', labels="cell_type")

#из статьи phenograph и louvain
data = np.arcsinh((data-1)/5)
communities, graph, Q = phenograph.cluster(data.values)
tic = time.clock()
nxgraph = nx.Graph(graph)
louvain_communities = community.best_partition(nxgraph)
louvainpd = pd.DataFrame(pd.Series(louvain_communities))
louvainpd.columns = ['community']
louvainpd['node'] = louvainpd.index
louvainpd = pd.DataFrame(louvainpd.groupby("community").count()).sort_values(by=['node'], ascending=False)
louvainpd['community'] = louvainpd.index
toc = time.clock()
print(toc-tic)
with open('phenocommunities.pickle', 'wb') as handle:
    pickle.dump(communities, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('louvaincommunities.pickle', 'wb') as handle:
    pickle.dump(louvain_communities, handle, protocol=pickle.HIGHEST_PROTOCOL)

#из статьи tSNE    
start_time = time.time()
raw_embedded = TSNE(n_components=2).fit_transform(data.values)
transposed = raw_embedded.transpose()
x = transposed[0]
y = transposed[1]

#из статьи импорт результатов phenograph и louvain 
infile = open('phenocommunities.pickle','rb')
phenopd = pickle.load(infile)
infile.close()
infile = open('louvaincommunities.pickle','rb')
louvainpd = pickle.load(infile)
infile.close()
louvainlist = [louvainpd[i] for i in louvainpd.keys()]

#создаем график
plt.rcParams['figure.figsize'] = [30, 15]
fig, ax = plt.subplots(1, 3)

#создаем подграфик для tSNE
labelset = list(set(labels))
print(labelset)
colors = []
norm = Normalize(vmin=0, vmax=1)
cmap = plt.cm.get_cmap('Spectral')
for i in range(10):
    colors.append(cmap(norm(i)))
ax[0].scatter(x,y, s=2, c=[colors[labelset.index(i)] for i in labels])
ax[0].set_xlabel("TSNE 1")
ax[0].set_ylabel("TSNE 2")
handles = []
for i in range(10):
    handles.append(Patch(facecolor=colors[i]))
ax[0].legend(handles, labelset, loc='upper right', prop={'size': 5}, framealpha=0.5)
ax[0].set_title("TSNE of gated labels")

#создаем подграфик для Phenograph
labelset = list(set(communities))
colors = []
norm = Normalize(vmin=0, vmax=len(labelset)-1)
cmap = plt.cm.get_cmap('Spectral')
for i in range(len(labelset)):
    colors.append(cmap(norm(i)))
ax[1].scatter(x,y, s=2, c=[colors[i] for i in communities])
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
handles = []
for i in range(len(labelset)-1):
    handles.append(Patch(facecolor=colors[i]))
ax[1].legend(handles, labelset, loc='upper right', prop={'size': 7}, framealpha=0.5)
ax[1].set_title("Phenograph clusters plotted on TSNE space")

#создаем подграфик для Louvain
labelset = list(set(louvainlist))
colors = []
norm = Normalize(vmin=0, vmax=len(labelset)-1)
cmap = plt.cm.get_cmap('Spectral')
for i in range(len(labelset)):
    colors.append(cmap(norm(i)))
ax[2].scatter(x,y, s=2, c=[colors[i] for i in louvainlist])
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
handles = []
for i in range(len(labelset)-1):
    handles.append(Patch(facecolor=colors[i]))
plt.legend(handles, labelset, loc='upper right', prop={'size': 7}, framealpha=0.5)

#все рисуем и сохраняем в файл
ax[2].set_title("Louvain clusters plotted on TSNE space")
plt.savefig('res.png', dpi=400)
