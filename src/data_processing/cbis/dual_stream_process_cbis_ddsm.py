#
import csv
import pandas as pd
from tqdm import trange

dual_stream_path = r'/home/kemove/PycharmProjects/Mammo/datasets/input-csv-files/cbis/cbis_ddsm_ds.csv'
multi_data_data = pd.read_csv('/home/kemove/PycharmProjects/Mammo/datasets/input-csv-files/cbis/cbis_ddsm_mil.csv', sep=';')

with open(dual_stream_path, "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["ShortPath", "Views", "BIRADS", "Groundtruth", "FolderName"])

for i in trange(len(multi_data_data)):
    row = multi_data_data.iloc[i]
    path = row['Patient_Id']
    split = row['FolderName']
    birads = row['Assessment']
    views = row['Views'].split('+')
    gt = row['Groundtruth']
    lviews = ''
    rviews = ''
    L_views = ['LCC', 'LMLO']
    R_views = ['RCC', 'RMLO']
    for j in range(len(views)):
        if views[j] in L_views:
            L_views.remove(views[j])
            lviews += views[j] + '+'
        elif views[j] in R_views:
            R_views.remove(views[j])
            rviews += views[j] + '+'
    with open(dual_stream_path, "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if len(L_views) == 0:
            writer.writerow([path, lviews[:-1], birads, gt, split])
        elif len(R_views) == 0:
            writer.writerow([path, rviews[:-1], birads, gt, split])

dual_stream_data = pd.read_csv(dual_stream_path, sep=';')

