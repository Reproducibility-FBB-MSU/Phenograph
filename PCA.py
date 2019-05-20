import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import os
import pandas as pd

#соединение файлов в одну таблицу
flag = 0
for filename in os.listdir('files/'):
    print(filename)
    if flag == 0:
        flag = 1
        res = pd.read_csv('files/'+str(filename), sep='\t')
    else:
        ad = pd.read_csv('files/'+str(filename), sep='\t')
        res = res.join(ad.set_index('Genes'), on='Genes')
 
#транспонирование для PCA
res = res.transpose()
res = res[1:]

#удаляем Nan
res = res.loc[:, :34013]

#фильтрация клеток и генов (из статьи)
res = res.loc[res.sum(axis=1) >= 500] 
leng = res.shape[0] 
res = res.loc[:, res.sum(axis=0)/leng >= 0.005] 
for i in list(res.columns): 
    if variation(res[i]) < 1.2: 
        res = res.drop(columns=[i])

#PCA
pca = PCA(n_components=40)
pca_result = pca.fit_transform(res)
pca_result = pd.DataFrame(pca_result)

#для графиков, добавляем колонку с cells
ct = ['cells' for i in range(len(list(res[list(res.columns)[0]])))]
pca_result['cell_type'] = ct

#запись результата в файл
pca_result.to_csv('PCA_res.csv', index=False, header=True)
