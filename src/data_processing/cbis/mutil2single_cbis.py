import csv
import pandas as pd
from tqdm import trange

single_data_path = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/cbis_ddsm_sil.csv'
mutil_data = pd.read_csv('/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/MG_training_files_cbis-ddsm_multiinstance2.csv', sep=';')

with open(single_data_path, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["FolderName", "path",  "Views", "BIRADS", "Groundtruth", "Split"])

trainortest = {'Training':'training', 'Test': 'test'}
for i in trange(len(mutil_data)):
    row = mutil_data.iloc[i]
    FolderName = row['FolderName']
    path = row['Patient_Id']
    birads = row['Assessment']
    view = row['Views'].split('+')
    Split = trainortest[row['FolderName'].split('_')[0].split('-')[1]]
    for j in range(len(view)):
        if birads >= 3:
            gt = 'malignant'
        else:
            gt = 'benign'
        with open(single_data_path, "a+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([FolderName, path, view[j], birads, gt, Split])
