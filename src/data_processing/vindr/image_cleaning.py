# -*- coding: utf-8 -*-
import csv
import os
import re
import cv2
import glob
import png
import numpy as np
import pandas as pd
import pydicom as dicom
import gzip
import matplotlib.cm as cm
from matplotlib import pyplot as plt


def load_dicom(filename):
    """ Load a dicom file. If it is compressed, it is unzipped first. """
    if (filename.endswith('.dcm')):
        ds = dicom.dcmread(filename)
    else:
        with gzip.open(filename) as fd:
            ds = dicom.dcmread(fd, force=True)
    return ds


def image_preprocessing(input_path, modality, path_to_input_csvfile, output_folder_path):
    # ============= Parameters =====================
    border_size = 105  # Border size
    blur = 21
    mask_dilate_iter = 20
    mask_erode_iter = 20
    sigma = 0.33
    c_study = 0
    df_img = {}

    # ============= input ==================
    df = pd.read_csv(path_to_input_csvfile)
    study_total = df[~df['study_id'].isnull()].shape[0]
    index_list = df.loc[~df['study_id'].isnull()].index
    end_file = study_total

    # ======= preprocess image ==========
    for i in range(0, end_file):
        print("image number:{}/{}".format(i, study_total))
        row = index_list[i]
        image = df.loc[row, 'path'] + '/' + df.loc[row, 'view'] + '.png'
        img_path = input_path + image
        print("image_path:", img_path)
        breast_side = df.loc[row, 'view'][0]

        c_image = 1
        # -- Read image -----------------------------------------------------------------------
        img = cv2.imread(img_path, -1)
        print("original image:", img.shape)
        print("original image dtype:", img.dtype)
        height, width = img.shape

        img_copy = img.copy()
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        print("img8 shape:", gray.shape)
        print("img8 dtype:", gray.dtype)

        # == Processing =======================================================================
        if modality == 'MG':
            # masking method
            foldname = output_folder_path + '/' + image.split('/')[0]
            if not os.path.exists(foldname):
                os.mkdir(foldname)
            filename = output_folder_path + '/' + png_root
            print("output_path:", filename)
            png_name = filename

            processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
                                                              mask_erode_iter, blur, border_size, breast_side)

            processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)

            print("img16 shape:", processed_img16.shape)
            print("img16 dtype:", processed_img16.dtype)

            # saving 16 bit images; if you need to save tthe 8 bit images then replace processed_img16 with processed_img8 in the png.Writer below and
            # change bitdepth from 16 to 8.

            with open(png_name, 'wb') as f:
                writer = png.Writer(
                    height=processed_img16.shape[0],
                    width=processed_img16.shape[1],
                    bitdepth=16,
                    greyscale=True
                )
                writer.write(f, processed_img16.tolist())

        c_study += 1

    # df_img_pd = pd.DataFrame.from_dict(df_img, orient='index',
    #                                    columns=['ImageName', 'pro_min_x', 'pro_min_y', 'pro_max_x', 'pro_max_y',
    #                                             'ori_height', 'ori_width', 'processed_height', 'processed_width'])
    # df_img_pd.to_csv(path_to_img_size, sep=';', na_rep='NULL', index=False)


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
    print("crop img:", crop_img.shape)
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


def read_imgsize_csvfile(path_to_img_size):
    df = pd.read_csv(path_to_img_size, sep=';')
    df['BreastSide'] = df['ImageName'].str.split('_').str[3].map({'LEFT': 'L', 'RIGHT': 'R'})
    df['pro_max_x'] = df.apply(lambda x: x['pro_max_x'] + 20 if x['BreastSide'] == 'L' else x['pro_max_x'], axis=1)
    df['pro_min_x'] = df.apply(lambda x: x['pro_min_x'] - 20 if x['BreastSide'] == 'R' else x['pro_min_x'], axis=1)
    # df.to_csv(path_to_img_size1, sep=';', na_rep='NULL', index=False)


# Initialization
modality = 'MG'

input_path = r'F:\data\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0\images/'  # <path-to-input-original-png-folder>

# this is a csv file with the following structure: AccessionNum;StudyDate;Groundtruth;Modality;StudyInstanceUID;StoragePath;dicom_processed;png_processed;Views
# This is a file containing each study, the location of the images belonging to that study, the groundtruth, the number of views in that study. This file will be passed as input to the image_preprocessing function
path_to_input_csvfile = r'/media/kemove/Elements/data/mammo.csv'  # <input csv file path containing list of data instances in the png folder>
# path_to_output_csvfile = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/vindr/vindr_gt.csv'  # <output csv file path>
# path for output folder - png files
output_folder_path_cleanedimages = r'/media/kemove/Elements/data/private_mammo/clean_data' #<output path to cleaned/preprocessed images>

# image_preprocessing based on modality
image_preprocessing(input_path, modality, path_to_input_csvfile, output_folder_path_cleanedimages)
