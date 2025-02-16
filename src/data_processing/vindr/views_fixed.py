# — coding: utf-8 –
import pandas as pd
from tqdm import trange

data = pd.read_csv(r'/datasets/input-csv-files/vindr/MG_training_files_vindr_multiinstance.csv', sep=';')

for i in trange(len(data)):
    views = data.iloc[i]['Views'].split('+')
    brs = data.iloc[i]['BreastDensity'].split(',')
    bis = data.iloc[i]['BIRADS'].split(',')
    br = bi = ''

    index1 = views.index('LCC')
    index2 = views.index('LMLO')
    index3 = views.index('RCC')
    index4 = views.index('RMLO')
    new_oreder = [index1, index2, index3, index4]
    brs = [brs[j] for j in new_oreder]
    bis = [bis[j] for j in new_oreder]
    for k in range(len(brs)):
        br += brs[k] + ','
        bi += bis[k] + ','

    data.loc[i, 'Views'] = 'LCC+LMLO+RCC+RMLO'
    data.loc[i, 'BreastDensity'] = br[:-1]
    data.loc[i, 'BIRADS'] = bi[:-1]

data.to_csv(r'D:\Code\Python_Code\Mammo\datasets\input-csv-files\vindr\vindr_mil.csv', index=False, sep=';')
data1 = pd.read_csv(r'/datasets/input-csv-files/vindr/vindr_mil.csv', sep=';')

