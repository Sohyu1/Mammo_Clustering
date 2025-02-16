import re
import os
import png
import glob
import pydicom
import pandas as pd
import pickle
import numpy as np
from multiprocessing import Pool
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import trange


def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth_output=12):
    """
    Save your mammogram from dicom format with ds.BitsStored bit as rescaled bitdepth_output png.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth output: what is the bitdepth of the output image you want!
    """
    try:
        ds = pydicom.dcmread(dicom_filename)
        image = ds.pixel_array
        image = apply_voi_lut(image, ds, index = 0)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            image = 2**ds.BitsStored - 1 - image
        if bitdepth_output == 16:
            image = np.uint16(image)
        with open(png_filename, 'wb') as f:
            writer = png.Writer(
                height=image.shape[0],
                width=image.shape[1],
                bitdepth=ds.BitsStored,
                greyscale=True
            )
            writer.write(f, image.tolist())
    except Exception as e:
        print(e)
        print(dicom_filename)

# def dicom_list_func(i):
#     data_root = r'F:\data\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0\images/'
#     out_root = r'D:\Code\Python_Code\Mammo\datasets\test_origin/'
#     path_to_dicom = r"F:\data\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0\finding_annotations.csv"
#     df_csv = pd.read_csv(path_to_dicom)
#     df = df_csv[(df_csv['split'] == 'test') & (df_csv['laterality'] == 'L')]
#     data_test_nonan = df.dropna(subset=['xmin'])
#     path_dicom = data_root + data_test_nonan.iloc[i]['study_id'] + '/' + data_test_nonan.iloc[i]['image_id'] + '.dicom'
#     case_path = out_root + df_csv.iloc[i]['study_id']
#     if not os.path.exists(case_path):
#         os.mkdir(case_path)
#     png_filename = case_path + '/' + df_csv.iloc[i]['laterality'] + df_csv.iloc[i]['view_position'] + '.png'
#     if not os.path.exists(png_filename):
#         save_dicom_image_as_png(path_dicom, png_filename, 16)


def dicom_list_func(data_test_nonan):
    data_root = r'F:\data\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0\images'
    out_root = r'D:\Code\Python_Code\Mammo\experiment\visualized_images\test_origin'
    for i in trange(len(data_test_nonan)):
        path_dicom = data_root + '\\' + data_test_nonan.iloc[i]['study_id'] + '\\' + data_test_nonan.iloc[i]['image_id'] + '.dicom'
        case_path = out_root + '\\' + data_test_nonan.iloc[i]['study_id']
        if not os.path.exists(case_path):
            os.mkdir(case_path)
        png_filename = case_path + '\\' + data_test_nonan.iloc[i]['laterality'] + data_test_nonan.iloc[i]['view_position'] + '.png'
        if not os.path.exists(png_filename):
            save_dicom_image_as_png(path_dicom, png_filename, 16)

if __name__ == '__main__':
    df_csv = pd.read_csv(r'F:\data\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0\finding_annotations.csv')
    # dicom_list = []
    # for i in trange(len(df_csv)):
    #     dicom_list.append(i)
    # # dicom_list = os.listdir(path_to_dicom)
    # #add the below line for CBIS to keep only the folders with mammogram images and exclude the folders with ROIs.
    # #dicom_list = list(filter(lambda x: not re.search('_\d$',x), dicom_list))
    # # dicom_list1 = []
    # # for idx, x, in enumerate(dicom_list):
    # #     dicom_list1.append([idx, x])
    # # dicom_list_func(dicom_list)
    # p = Pool(10)
    # p.CNN(dicom_list_func, dicom_list)

    # test gt
    dicom_list = []
    # df = df_csv[(df_csv['split'] == 'test') & (df_csv['laterality'] == 'L')]
    df = df_csv[df_csv['split'] == 'test']
    data_test_nonan = df.dropna(subset=['xmin'])
    dicom_list_func(data_test_nonan)
    # for i in trange(len(data_test_nonan)):
    #     dicom_list.append(i)
    # p = Pool(10)
    # p.CNN(dicom_list_func, dicom_list)