import os
import re
import shutil
import csv
import cv2
import glob
import png
import numpy as np
import pandas as pd
import pydicom as dicom
import gzip
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib import pyplot as plt

data = pd.read_csv(
    r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/MG_training_files_cbis-ddsm_multiinstance2.csv',
    sep=';')

view2views = ['LCC', 'LMLO', 'RCC', 'RMLO']
sum = 0
for i in range(len(data)):
    view = data.iloc[i]['Views'].split('+')
    for j in range(len(view2views)):
        if view2views[j] not in view:
            sum += 1
            break

res = len(data) - sum


path_root = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data'
path = os.listdir(path_root)

single_views_left = ['LCC.png', 'LMLO.png']
single_views_right = ['RCC.png', 'RMLO.png']

sum_left = 0
sum_right = 0
for i in range(len(path)):
    png_files = os.listdir(path_root + '/' + path[i])
    if single_views_left[0] in png_files and single_views_left[1] in png_files:
        sum_left += 1
    if single_views_right[0] in png_files and single_views_right[1] in png_files:
        sum_right += 1

for i in range(len(data)):
    view = data.iloc[i]['Views'].split('+')
    new_view = list({}.fromkeys(view).keys())
    str = ''
    for j in range(len(new_view)):
        str += new_view[j] + '+'
    data.at[i, 'Views'] = str[:-1]

data.to_csv('/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/cbis_ddsm_mil.csv', index=False, sep=';')


