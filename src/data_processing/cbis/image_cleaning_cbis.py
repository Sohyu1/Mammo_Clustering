# -*- coding: utf-8 -*-
import os
import re
import shutil

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


# def load_dicom(filename):
#     "' Load a dicom file. If it is compressed, it is unzipped first. '"
#     # print("filename:",filename)
#     if (filename.endswith('.dcm')):
#         ds = dicom.dcmread(filename)
#     else:
#         with gzip.open(filename) as fd:
#             ds = dicom.dcmread(fd, force=True)
#     # print(ds)
#     return ds


# def image_preprocessing(start_file, end_file, selected_image, input_path, modality, path_to_input_csvfile,
#                         output_folder_path):
#     # ============= Parameters =====================
#     border_size = 105  # Border size
#     blur = 21
#     mask_dilate_iter = 20
#     mask_erode_iter = 20
#     sigma = 0.33
#     c_study = 0
#     df_img = {}
#
#     # ============= input ==================
#     # df = pd.read_csv(path_to_input_csvfile, sep=';')
#     df = pd.read_csv(path_to_input_csvfile)
#     # study_total = df[~df['ImageName'].isnull()].shape[0]
#     # index_list = df.loc[~df['ImageName'].isnull()].index
#     study_total = df[df['Series Description'] == 'full mammogram images'].shape[0]
#     index_list = df.loc[df['Series Description'] == 'full mammogram images'].index
#     print(study_total)
#     if selected_image != '':
#         start_file = df[df['ImageName'].str.strip('_1.1') == selected_image].index[0]
#         end_file = df[df['ImageName'].str.strip('_1.1') == selected_image].index[0] + 1
#         print(start_file)
#     else:
#         if end_file == 0:
#             end_file = study_total
#
#     # ======= preprocess image ==========
#     for i in range(start_file, end_file):
#         print("image number:{}/{}".format(i, study_total))
#         row = index_list[i]
#         img_name = df.loc[row, 'Study UID'].split('/')
#         img_path = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/data/' + img_name[0]
#         png_root =  df.loc[row, 'Laterality'] + df.loc[row, 'PatientOrientation'] + '.png'
#         print("image_path:", img_path)
#         # breast_side = df.loc[row, 'Views'][0]
#         breast_side = df.loc[row, 'Laterality']
#
#         c_image = 1
#         # -- Read image -----------------------------------------------------------------------
#         img = cv2.imread(img_path, -1)
#         print("original image:", img.shape)
#         print("original image dtype:", img.dtype)
#         try:
#             height, width = img.shape
#             # print(height, width)
#         except:
#             out1 = open('./images_not_processed_ori_image_empty_MG.txt', 'a')
#             out1.write(img_path + '\n')
#             out1.close()
#             continue
#
#         img_copy = img.copy()
#         gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#         print("img8 shape:", gray.shape)
#         print("img8 dtype:", gray.dtype)
#
#         # == Processing =======================================================================
#         if modality == 'MG':
#             # masking method
#             foldname = output_folder_path + '/' + img_name[2]
#             if not os.path.exists(foldname):
#                 os.mkdir(foldname)
#             # filename = output_folder_path + '/' + image
#             filename = output_folder_path + '/' + png_root
#             print("output_path:", filename)
#             processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
#                                                               mask_erode_iter, blur, border_size, breast_side)
#
#             df_img[img_name[2]] = [img_name[2] + '/' + img_name[3], x, y, x + w, y + h, img.shape[0], img.shape[1],
#                                            processed_img8.shape[0], processed_img8.shape[1]]
#
#             processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)
#             print("img16 shape:", processed_img16.shape)
#             print("img16 dtype:", processed_img16.dtype)
#             plt.imsave(filename, processed_img16, cmap='gray')
#             # with open(filename, 'wb') as f:
#             #         writer = png.Writer(
#             #             height=processed_img16.shape[0],
#             #             width=processed_img16.shape[1],
#             #             bitdepth=16,
#             #             greyscale=True
#             #         )
#             #         writer.write(f, processed_img16.tolist())
#
#
#         c_study += 1
#
#     df_img_pd = pd.DataFrame.from_dict(df_img, orient='index',
#                                        columns=['ImageName', 'pro_min_x', 'pro_min_y', 'pro_max_x', 'pro_max_y',
#                                                 'ori_height', 'ori_width', 'processed_height', 'processed_width'])
#     df_img_pd.to_csv(path_to_img_size, sep=';', na_rep='NULL', index=False)


def mask_image(disp_id, gray, img, sigma, MASK_DILATE_ITER, MASK_ERODE_ITER, BLUR, border_size, breast_side):
    '''cv2.namedWindow("Display frame"+str(disp_id), cv2.WINDOW_NORMAL)
    cv2.imshow("Display frame"+str(disp_id),gray)
    cv2.waitKey(0)
    '''
    # -- Edge detection -------------------------------------------------------------------
    v = np.median(gray)

    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # for zgt, vindr
    # edges = cv2.Canny(gray, lower, upper)

    # for cbis
    edges = cv2.Canny(gray, 0, 10)

    # edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    edges = edges[border_size:-border_size, border_size:-border_size]
    edges = cv2.GaussianBlur(edges, (BLUR, BLUR), 0)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    # for cbis (this line not used for vindr, zgt dataset)
    edges = cv2.copyMakeBorder(edges, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None,
                               value=0)

    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), None, iterations=MASK_ERODE_ITER)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), None, iterations=MASK_DILATE_ITER)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), None, iterations=MASK_DILATE_ITER)
    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), None, iterations=MASK_ERODE_ITER)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for idx, c in enumerate(contours):
        contour_info.append((
            idx,
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[3], reverse=True)
    max_contour = contour_info[0][1]

    cimg = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(cimg, [max_contour], -1, color=(255, 255, 255), thickness=-1)
    '''cv2.imshow("Display frame"+str(disp_id),cimg)
    cv2.waitKey(0)
    '''
    cimg = cv2.dilate(cimg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), None, iterations=MASK_DILATE_ITER)
    '''cv2.imshow("Display frame"+str(disp_id),cimg)
    cv2.waitKey(0)
    '''
    res = cv2.bitwise_and(img, img, mask=cimg)
    '''print(res.shape)
    cv2.imshow("Display frame"+str(disp_id),res)
    cv2.waitKey(0)
    '''
    x, y, w, h = cv2.boundingRect(res[:, :])
    if breast_side == 'L':
        crop_img = res[y:y + h, x:min(x + w + 20, res.shape[1])]
    elif breast_side == 'R':
        crop_img = res[y:y + h, max(0, x - 20):x + w]
    else:
        crop_img = res[y:y + h, x:x + w]
    # print(gray.shape)
    # print("crop img:", crop_img.shape)
    '''cv2.imshow("Display frame"+str(disp_id),crop_img)
    cv2.waitKey(0)
    '''
    return crop_img, cimg, x, y, w, h


def image_16bit_preprocessing(img16, img_mask, x, y, w, h, breast_side):
    res = cv2.bitwise_and(img16, img16, mask=img_mask)
    if breast_side == 'L':
        res_crop = res[y:y + h, x:x + w + 20]
    elif breast_side == 'R':
        res_crop = res[y:y + h, x - 20:x + w]
    else:
        res_crop = res[y:y + h, x:x + w]
    return res_crop


def read_imgsize_csvfile(path_to_img_size, path_to_img_size1):
    df = pd.read_csv(path_to_img_size, sep=';')
    df['BreastSide'] = df['ImageName'].str.split('_').str[3].map({'LEFT': 'L', 'RIGHT': 'R'})
    df['pro_max_x'] = df.apply(lambda x: x['pro_max_x'] + 20 if x['BreastSide'] == 'L' else x['pro_max_x'], axis=1)
    df['pro_min_x'] = df.apply(lambda x: x['pro_min_x'] - 20 if x['BreastSide'] == 'R' else x['pro_min_x'], axis=1)
    df.to_csv(path_to_img_size1, sep=';', na_rep='NULL', index=False)


# Initialization
# modality = 'MG'
#
# input_path = r'/media/kemove/杨翻盖/data/ddsm/csv/dicom_info.csv'  # <path-to-input-original-png-folder>
#
# # this is a csv file with the following structure: AccessionNum;StudyDate;Groundtruth;Modality;StudyInstanceUID;StoragePath;dicom_processed;png_processed;Views
# # This is a file containing each study, the location of the images belonging to that study, the groundtruth, the number of views in that study. This file will be passed as input to the image_preprocessing function
# path_to_input_csvfile = r'/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/metadata.csv'  # <input csv file path containing list of data instances in the png folder>
# path_to_output_csvfile = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/MIL_gt.csv'  # <output csv file path>
# path_to_img_size = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/MIL_co.csv'  # <output csv file to store the preprocessed image coordinates>
# path_to_img_size1 = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/MIL_co1.csv'  # <output csv file to store the preprocessed image coordinates after padding adjustment>

# path for output folder - png files
output_folder_path_cleanedimages = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data'  # <output path to cleaned/preprocessed images>

start = 0
end = 0
selected_image = ''

# image_preprocessing based on modality
# image_preprocessing(start, end, selected_image, input_path, modality, path_to_input_csvfile,
#                     output_folder_path_cleanedimages)
# read_imgsize_csvfile(path_to_img_size)

df_csv = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/metadata.csv')
df = df_csv[df_csv['Series Description'] == 'full mammogram images']
mass_csv_train = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/mass_case_description_train_set.csv')
mass_csv_test = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/mass_case_description_test_set.csv')
calc_csv_train = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/calc_case_description_train_set.csv')
calc_csv_test = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/calc_case_description_test_set.csv')

border_size = 105  # Border size
blur = 21
mask_dilate_iter = 20
mask_erode_iter = 20
sigma = 0.33
c_study = 0
c_image = 1

jpg_path = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/data'
views = {
    'LEFTCC': 'LCC',
    'LEFTMLO': 'LMLO',
    'RIGHTCC': 'RCC',
    'RIGHTMLO': 'RMLO',
         }
breast_side_two = {
    'LEFT': 'L',
    'RIGHT': 'R',
         }

for i in tqdm(range(len(mass_csv_train))):
    img_root = mass_csv_train['image file path'][i].split('/')[1]
    # mass_csv_train['image file path'][i].split('/')[0].split('_')
    img_original_path = jpg_path + '/' + img_root
    view = mass_csv_train['image file path'][i].split('/')[0].split('_')[3] + mass_csv_train['image file path'][i].split('/')[0].split('_')[4]
    img_paths = os.listdir(img_original_path)
    breast_side = breast_side_two[mass_csv_train['image file path'][i].split('/')[0].split('_')[3]]
    for j in range(len(img_paths)):
        img = cv2.imread(img_original_path + '/' + img_paths[j], -1)
        file_path = img_original_path + '/' + img_paths[j]
        img_copy = img.copy()
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # masking method
        foldname = output_folder_path_cleanedimages + '/' + img_root
        if not os.path.exists(foldname):
            os.mkdir(foldname)
        # filename = output_folder_path + '/' + image
        filename = output_folder_path_cleanedimages + '/' + img_root + '/' + views[view] + '.png'
        if os.path.exists(filename):
            continue
        processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
                                                          mask_erode_iter, blur, border_size, breast_side)

        processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)
        with open(filename, 'wb') as f:
            writer = png.Writer(
                height=processed_img16.shape[0],
                width=processed_img16.shape[1],
                bitdepth=16,
                greyscale=True
            )
            writer.write(f, processed_img16.tolist())

for i in tqdm(range(len(mass_csv_train))):
    img_root = mass_csv_train['image file path'][i].split('/')[1]
    file_root = mass_csv_train['image file path'][i].split('/')[0].split('_')[1]+'_'+mass_csv_train['image file path'][i].split('/')[0].split('_')[2]
    file_name = mass_csv_train['image file path'][i].split('/')[0].split('_')[3][0]+mass_csv_train['image file path'][i].split('/')[0].split('_')[4] + '.png'
    moveto = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/' + file_root
    if not os.path.exists(moveto):
        os.mkdir(moveto)
    move2path = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data/' + img_root + '/' + file_name
    if not os.path.exists(move2path):
        shutil.move(move2path, '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/'+file_root)

#
for i in tqdm(range(len(mass_csv_test))):
    img_root = mass_csv_test['image file path'][i].split('/')[1]
    mass_csv_test['image file path'][i].split('/')[0].split('_')
    img_original_path = jpg_path + '/' + img_root
    view = mass_csv_test['image file path'][i].split('/')[0].split('_')[3] + mass_csv_test['image file path'][i].split('/')[0].split('_')[4]
    img_paths = os.listdir(img_original_path)
    breast_side = breast_side_two[mass_csv_test['image file path'][i].split('/')[0].split('_')[3]]
    for j in range(len(img_paths)):
        img = cv2.imread(img_original_path + '/' + img_paths[j], -1)
        file_path = img_original_path + '/' + img_paths[j]
        img_copy = img.copy()
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # masking method
        foldname = output_folder_path_cleanedimages + '/' + img_root
        if not os.path.exists(foldname):
            os.mkdir(foldname)
        # filename = output_folder_path + '/' + image
        filename = output_folder_path_cleanedimages + '/' + img_root + '/' + views[view] + '.png'
        if os.path.exists(filename):
            continue
        processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
                                                          mask_erode_iter, blur, border_size, breast_side)

        processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)
        # plt.imsave(filename, processed_img16, cmap='gray')
        with open(filename, 'wb') as f:
            writer = png.Writer(
                height=processed_img16.shape[0],
                width=processed_img16.shape[1],
                bitdepth=16,
                greyscale=True
            )
            writer.write(f, processed_img16.tolist())
for i in tqdm(range(len(mass_csv_test))):
    img_root = mass_csv_test['image file path'][i].split('/')[1]
    file_root = mass_csv_test['image file path'][i].split('/')[0].split('_')[1]+'_'+mass_csv_test['image file path'][i].split('/')[0].split('_')[2]
    file_name = mass_csv_test['image file path'][i].split('/')[0].split('_')[3][0]+mass_csv_test['image file path'][i].split('/')[0].split('_')[4] + '.png'
    moveto = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/' + file_root
    if not os.path.exists(moveto):
        os.mkdir(moveto)
    move2path = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data/' + img_root + '/' + file_name
    if not os.path.exists(move2path):
        shutil.move(move2path,
                    '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/' + file_root) #

for i in tqdm(range(len(calc_csv_train))):
    img_root = calc_csv_train['image file path'][i].split('/')[1]
    calc_csv_train['image file path'][i].split('/')[0].split('_')
    img_original_path = jpg_path + '/' + img_root
    view = calc_csv_train['image file path'][i].split('/')[0].split('_')[3] + calc_csv_train['image file path'][i].split('/')[0].split('_')[4]
    img_paths = os.listdir(img_original_path)
    breast_side = breast_side_two[calc_csv_train['image file path'][i].split('/')[0].split('_')[3]]
    for j in range(len(img_paths)):
        img = cv2.imread(img_original_path + '/' + img_paths[j], -1)
        file_path = img_original_path + '/' + img_paths[j]
        img_copy = img.copy()
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # masking method
        foldname = output_folder_path_cleanedimages + '/' + img_root
        if not os.path.exists(foldname):
            os.mkdir(foldname)
        # filename = output_folder_path + '/' + image
        filename = output_folder_path_cleanedimages + '/' + img_root + '/' + views[view] + '.png'
        if os.path.exists(filename):
            continue
        processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
                                                          mask_erode_iter, blur, border_size, breast_side)

        processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)
        # plt.imsave('/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/test/1.png', processed_img16, cmap='gray')
        with open(filename, 'wb') as f:
            writer = png.Writer(
                height=processed_img16.shape[0],
                width=processed_img16.shape[1],
                bitdepth=16,
                greyscale=True
            )
            writer.write(f, processed_img16.tolist())

for i in tqdm(range(len(calc_csv_train))):
    img_root = calc_csv_train['image file path'][i].split('/')[1]
    file_root = calc_csv_train['image file path'][i].split('/')[0].split('_')[1]+'_'+calc_csv_train['image file path'][i].split('/')[0].split('_')[2]
    file_name = calc_csv_train['image file path'][i].split('/')[0].split('_')[3][0]+calc_csv_train['image file path'][i].split('/')[0].split('_')[4] + '.png'
    moveto = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/' + file_root
    if not os.path.exists(moveto):
        os.mkdir(moveto)
    move2path = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data/' + img_root + '/' + file_name
    if not os.path.exists(move2path):
        shutil.move(move2path,
                    '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/' + file_root)


for i in tqdm(range(len(calc_csv_test))):
    img_root = calc_csv_test['image file path'][i].split('/')[1]
    calc_csv_test['image file path'][i].split('/')[0].split('_')
    img_original_path = jpg_path + '/' + img_root
    view = calc_csv_test['image file path'][i].split('/')[0].split('_')[3] + calc_csv_test['image file path'][i].split('/')[0].split('_')[4]
    img_paths = os.listdir(img_original_path)
    breast_side = breast_side_two[calc_csv_test['image file path'][i].split('/')[0].split('_')[3]]
    for j in range(len(img_paths)):
        img = cv2.imread(img_original_path + '/' + img_paths[j], -1)
        file_path = img_original_path + '/' + img_paths[j]
        img_copy = img.copy()
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # masking method
        foldname = output_folder_path_cleanedimages + '/' + img_root
        if not os.path.exists(foldname):
            os.mkdir(foldname)
        # filename = output_folder_path + '/' + image
        filename = output_folder_path_cleanedimages + '/' + img_root + '/' + views[view] + '.png'
        if os.path.exists(filename):
            continue
        processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
                                                          mask_erode_iter, blur, border_size, breast_side)

        processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)
        plt.imsave(filename, processed_img16, cmap='gray')
        with open(filename, 'wb') as f:
            writer = png.Writer(
                height=processed_img16.shape[0],
                width=processed_img16.shape[1],
                bitdepth=16,
                greyscale=True
            )
            writer.write(f, processed_img16.tolist())

for i in tqdm(range(len(calc_csv_test))):
    img_root = calc_csv_test['image file path'][i].split('/')[1]
    file_root = calc_csv_test['image file path'][i].split('/')[0].split('_')[1]+'_'+calc_csv_test['image file path'][i].split('/')[0].split('_')[2]
    file_name = calc_csv_test['image file path'][i].split('/')[0].split('_')[3][0]+calc_csv_test['image file path'][i].split('/')[0].split('_')[4] + '.png'
    moveto = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/' + file_root
    if not os.path.exists(moveto):
        os.mkdir(moveto)
    move2path = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data/' + img_root + '/' + file_name
    if not os.path.exists(move2path):
        shutil.move(move2path,
                    '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/' + file_root)


# breast_side = 'L'
# a_path = '/media/kemove/杨翻盖/data/ddsm/jpeg/1.3.6.1.4.1.9590.100.1.2.374115997511889073021386151921807063992/1-043.jpg'
# img = cv2.imread(a_path, -1)
# img_copy = img.copy()
# gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
# b_path = '/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data/1.3.6.1.4.1.9590.100.1.2.374115997511889073021386151921807063992'
# if not os.path.exists(b_path):
#     os.mkdir(b_path)
# processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
#                                                           mask_erode_iter, blur, border_size, breast_side)
# processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)
# plt.imsave('/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data/1.3.6.1.4.1.9590.100.1.2.374115997511889073021386151921807063992/LCC.png', processed_img16, cmap='gray')