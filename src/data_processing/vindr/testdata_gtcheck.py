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


def mask_image(disp_id, gray, img, sigma, MASK_DILATE_ITER, MASK_ERODE_ITER, BLUR, border_size, breast_side, image):
    """
    cv2.namedWindow("Display frame"+str(disp_id), cv2.WINDOW_NORMAL)
    cv2.imshow("Display frame"+str(disp_id),gray)
    cv2.waitKey(0)
    """
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

    edges = edges[border_size:-border_size, border_size:-border_size]
    edges = cv2.GaussianBlur(edges, (BLUR, BLUR), 0)

    # for cbis (this line not used for vindr, zgt dataset)
    edges = cv2.copyMakeBorder(edges, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value=0)

    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), None, iterations=MASK_ERODE_ITER)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), None, iterations=MASK_DILATE_ITER)

    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), None, iterations=MASK_DILATE_ITER)
    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), None, iterations=MASK_ERODE_ITER)

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

    cimg = cv2.dilate(cimg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), None, iterations=MASK_DILATE_ITER)

    res = cv2.bitwise_and(img, img, mask=cimg)

    x, y, w, h = cv2.boundingRect(res[:, :])
    if breast_side == 'L':
        crop_img = res[y:y + h, x:min(x + w + 20, res.shape[1])]
        with open(r"D:\Code\Python_Code\Mammo\experiment\visualized_images\test_crop.csv", "a+", newline='') as csvfile1:
            writer1 = csv.writer(csvfile1)
            writer1.writerow([image, x, min(x + w + 20, res.shape[1]), y, y + h, h, w])
    elif breast_side == 'R':
        crop_img = res[y:y + h, max(0, x - 20):x + w]
        with open(r"D:\Code\Python_Code\Mammo\experiment\visualized_images\test_crop.csv", "a+", newline='') as csvfile1:
            writer1 = csv.writer(csvfile1)
            writer1.writerow([image, max(0, x - 20), x + w, y, y + h, h, w])
    else:
        crop_img = res[y:y + h, x:x + w]

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


def image_preprocessing(input_path, path_to_input_csvfile, output_folder_path):
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
    df = df[df['split'] == 'test']
    df = df.dropna(subset=['xmin'])
    study_total = df[~df['study_id'].isnull()].shape[0]
    index_list = df.loc[~df['study_id'].isnull()].index
    end_file = study_total

    # ======= preprocess image ==========
    for i in range(0, end_file):
        print("image number:{}/{}".format(i, study_total))
        row = index_list[i]
        png_root = df.loc[row, 'laterality'] + df.loc[row, 'view_position'] + '.png'
        image = df.loc[row, 'study_id'] + '/' + png_root
        img_path = input_path + image
        print("image_path:", img_path)
        breast_side = df.loc[row, 'laterality']

        c_image = 1
        # -- Read image -----------------------------------------------------------------------
        img = cv2.imread(img_path, -1)
        # print("original image:", img.shape)
        # print("original image dtype:", img.dtype)
        height, width = img.shape

        img_copy = img.copy()
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # print("img8 shape:", gray.shape)
        # print("img8 dtype:", gray.dtype)

        # == Processing =======================================================================
        # masking method
        foldname = output_folder_path + '/' + image.split('/')[0]
        if not os.path.exists(foldname):
            os.mkdir(foldname)
        filename = foldname + '/' + png_root
        print("output_path:", filename)
        png_name = filename
        processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
                                                              mask_erode_iter, blur, border_size, breast_side, image)

        if not os.path.exists(png_name):

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

input_path = r'/experiment/visualized_images/test/test_origin/'
path_to_input_csvfile = r'F:\data\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0\finding_annotations.csv'

output_folder_path_cleanedimages = r'/experiment/visualized_images/test/test_clean'

with open(r"D:\Code\Python_Code\Mammo\experiment\visualized_images\test_crop.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "x1", "x2", "y1", "y2", "h", "w"])

image_preprocessing(input_path, path_to_input_csvfile, output_folder_path_cleanedimages)



