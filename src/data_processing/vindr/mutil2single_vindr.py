import csv
import pandas as pd
from tqdm import trange

single_data_path = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/vindr/vindr_sil.csv'
mutil_data = pd.read_csv('/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/vindr/MG_training_files_vindr_multiinstance.csv', sep=';')

with open(single_data_path, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["path", "Views", "BIRADS", "Groundtruth", "Split"])
for i in trange(len(mutil_data)):
    row = mutil_data.iloc[i]
    path = row['StudyInstanceUID']
    split = row['Split']
    birads = row['BIRADS'].split(',')
    for j in range(len(birads)):
        view = row['Views'].split('+')[j]
        if int(birads[j]) >= 3:
            gt = 'malignant'
        else:
            gt = 'benign'
        with open(single_data_path, "a+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([path, view, int(birads[j]), gt, split])
