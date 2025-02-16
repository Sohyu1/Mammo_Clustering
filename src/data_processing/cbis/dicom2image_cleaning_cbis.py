#
import os

import cv2
import png
import glob
import pydicom
import pandas as pd
import pickle
import numpy as np
from multiprocessing import Pool
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import trange


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


def save_dicom_image_as_png(dicom_filename, png_filename, breast_side):
    """
    Save your mammogram from dicom format with ds.BitsStored bit as rescaled bitdepth_output png.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth output: what is the bitdepth of the output image you want!
    """
    try:
        ds = pydicom.read_file(dicom_filename)
        image = ds.pixel_array
        image = apply_voi_lut(image, ds, index = 0)
        image = np.uint16(image)
        img_copy = image.copy()
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        processed_img8, img_mask, x, y, w, h = mask_image(c_image, gray, gray, sigma, mask_dilate_iter,
                                                          mask_erode_iter, blur, border_size, breast_side)

        processed_img16 = image_16bit_preprocessing(image, img_mask, x, y, w, h, breast_side)
        with open(png_filename, 'wb') as f:
            writer = png.Writer(
                height=processed_img16.shape[0],
                width=processed_img16.shape[1],
                bitdepth=ds.BitsStored,
                greyscale=True
            )
            writer.write(f, processed_img16.tolist())
    except Exception as e:
        print(e)
        print(dicom_filename)

def dicom_list_func(i):
    data_root = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data/'
    path_to_dicom = r"/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142"
    df = pd.read_csv(
        '/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/metadata.csv')
    df_csv = df[df['Series Description'] == 'full mammogram images']
    a = df_csv['Subject ID'].iloc[i].split('_')
    root_path = a[1] + '_' + a[2]
    breast_side = breast_side_two[a[3]]
    name = breast_side + a[4]
    path_dicom_file = path_to_dicom + df_csv['File Location'].iloc[i][1:]
    dicom_files = os.listdir(path_dicom_file)
    case_path = data_root + root_path
    if not os.path.exists(case_path):
        os.mkdir(case_path)
    for _, dicom_file in enumerate(dicom_files):
        path_dicom = path_dicom_file + '/' + dicom_file
        png_filename = case_path + '/' + name + '.png'
        if not os.path.exists(png_filename):
            save_dicom_image_as_png(path_dicom, png_filename, breast_side)



if __name__ == '__main__':
    border_size = 105  # Border size
    blur = 21
    mask_dilate_iter = 20
    mask_erode_iter = 20
    sigma = 0.33
    c_study = 0
    c_image = 1
    breast_side_two = {
        'LEFT': 'L',
        'RIGHT': 'R',
    }

    df = pd.read_csv(
        '/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/metadata.csv')
    df_csv = df[df['Series Description'] == 'full mammogram images']
    dicom_list = []
    for i in trange(len(df_csv)):
        dicom_list.append(i)
    p = Pool(10)
    p.map(dicom_list_func, dicom_list)
