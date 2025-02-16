import csv
import pandas as pd
from tqdm import trange

dual_stream_path = r'/home/kemove/PycharmProjects/Mammo/datasets/input-csv-files/vindr/vindr_ds.csv'
multi_data_data = pd.read_csv('/home/kemove/PycharmProjects/Mammo/datasets/input-csv-files/vindr/MG_training_files_vindr_multiinstance.csv', sep=';')

with open(dual_stream_path, "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["path", "Views", "BIRADS", "Groundtruth", "Split"])
for i in trange(len(multi_data_data)):
    row = multi_data_data.iloc[i]
    path = row['StudyInstanceUID']
    split = row['Split']
    birads = row['BIRADS'].split(',')
    views = row['Views'].split('+')
    lbirads = ''
    lviews = ''
    rbirads = ''
    rviews = ''
    L_views = ['LCC', 'LMLO']
    R_views = ['RCC', 'RMLO']
    for j in range(len(views)):
        if views[j] in L_views:
            L_views.remove(views[j])
            lbirads += birads[j] + ','
            lviews += views[j] + '+'
            if int(birads[j]) >= 3:
                lgt = 'malignant'
            else:
                lgt = 'benign'
        elif views[j] in R_views:
            R_views.remove(views[j])
            rbirads += birads[j] + ','
            rviews += views[j] + '+'
            if int(birads[j]) >= 3:
                rgt = 'malignant'
            else:
                rgt = 'benign'
    with open(dual_stream_path, "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([path, lviews[:-1], lbirads[:-1], lgt, split])
        writer.writerow([path, rviews[:-1], rbirads[:-1], rgt, split])

dual_stream_data = pd.read_csv(dual_stream_path, sep=';')

