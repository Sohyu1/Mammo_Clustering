import os
import csv
import glob
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

file_creation = 'MIL'


# ------------------------------------------------------------------MIL csv file creation----------------------------------------------#

# creating mammogram csv file
def create_MIL_csv_file(cbis_ddsm_trainingfile):
    # open the file in the write mode
    f = open(cbis_ddsm_trainingfile, 'w', newline='')
    # create the csv writer
    writer = csv.writer(f, delimiter=';')
    header = ['FolderName', 'Patient_Id', 'AbnormalityType', 'Views', 'BreastDensity', 'Assessment', 'Groundtruth', 'AssessmentMax', 'ShortPath']
    writer.writerow(header)
    four_views = {
        'LEFTCC': 'LCC',
        'LEFTMLO': 'LMLO',
        'RIGHTCC': 'RCC',
        'RIGHTMLO': 'RMLO',
    }
    mass_csv_train = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/mass_case_description_train_set.csv')
    mass_csv_test = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/mass_case_description_test_set.csv')
    calc_csv_train = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/calc_case_description_train_set.csv')
    calc_csv_test = pd.read_csv('/home/kemove/PycharmProjects/cbis/manifest-ZkhPvrLo5216730872708713142/calc_case_description_test_set.csv')
    output_folderlist = os.listdir(output_foldername)
    for i in range(len(output_folderlist)):
        StudyInstanceUID = output_folderlist[i]
        Patient_Id = output_folderlist[i]
        mtrain = mass_csv_train[mass_csv_train['patient_id'] == StudyInstanceUID]
        mtest = mass_csv_test[mass_csv_test['patient_id'] == StudyInstanceUID]
        ctrain = calc_csv_train[calc_csv_train['patient_id'] == StudyInstanceUID]
        ctest = calc_csv_test[calc_csv_test['patient_id'] == StudyInstanceUID]
        FolderName = ''
        views = ''
        bds = ''
        birads = ''
        gt = 'benign'
        print(i, '/', len(output_folderlist))
        if len(mtrain):
            for j in range(len(mtrain)):
                FolderName = mtrain.iloc[j]['image file path'].split('/')[0]
                AbnormalityType = 'Mass'
                views += four_views[mtrain.iloc[j]['left or right breast'] + mtrain.iloc[j]['image view']] + '+'
                bds = mtrain.iloc[j]['breast_density']
                birads = mtrain.iloc[j]['assessment']
                if mtrain.iloc[j]['pathology'] == 'MALIGNANT':
                    gt = 'malignant'
        if len(mtest):
            for j in range(len(mtest)):
                FolderName = mtest.iloc[j]['image file path'].split('/')[0]
                AbnormalityType = 'Mass'
                views += four_views[mtest.iloc[j]['left or right breast'] + mtest.iloc[j]['image view']] + '+'
                bds = mtest.iloc[j]['breast_density']
                birads = mtest.iloc[j]['assessment']
                if mtest.iloc[j]['pathology'] == 'MALIGNANT':
                    gt = 'malignant'
        if len(ctrain):
            for j in range(len(ctrain)):
                FolderName = ctrain.iloc[j]['image file path'].split('/')[0]
                AbnormalityType = 'Calc'
                views += four_views[ctrain.iloc[j]['left or right breast'] + ctrain.iloc[j]['image view']] + '+'
                bds = ctrain.iloc[j]['breast density']
                birads = ctrain.iloc[j]['assessment']
                if ctrain.iloc[j]['pathology'] == 'MALIGNANT':
                    gt = 'malignant'
        if len(ctest):
            for j in range(len(ctest)):
                FolderName = ctest.iloc[j]['image file path'].split('/')[0]
                AbnormalityType = 'Calc'
                views += four_views[ctest.iloc[j]['left or right breast'] + ctest.iloc[j]['image view']] + '+'
                bds = ctest.iloc[j]['breast density']
                birads = ctest.iloc[j]['assessment']
                if ctest.iloc[j]['pathology'] == 'MALIGNANT':
                    gt = 'malignant'
        row = [FolderName, Patient_Id, AbnormalityType, views[:-1], bds, birads, gt, birads, Patient_Id]
        writer.writerow(row)
    f.close()


def aggregating_patient_case_info(grp):
    # grp_agg=pd.DataFrame(columns=['FolderName', 'PatientId', 'AbnormalityType', 'Views', 'FullPath', 'BreastDensity', 'Assessment', 'Groundtruth'], index=range(1))
    grp_agg = pd.DataFrame(
        columns=['FolderName', 'PatientId', 'BreastDensity', 'Assessment', 'Groundtruth', 'AssessmentMax'],
        index=range(1))
    # if len(np.unique(grp['breast_density']))>1:
    #    print("yes")
    grp_agg['FolderName'] = np.unique(grp['foldername'])
    grp_agg['PatientId'] = np.unique(grp['patient_id'])
    try:
        grp_agg['BreastDensity'] = np.unique(grp['breast_density'])
    except:
        grp_agg['BreastDensity'] = np.unique(grp['breast density'])
    # if len(np.unique(grp['assessment']))>1:
    #    print(list(np.unique(grp['assessment'])))
    #    print(grp[['patient_id','assessment','left or right breast','image view', 'abnormality type','subtlety']])
    grp_agg['Assessment'] = ",".join(str(i) for i in list(np.unique(grp['assessment'])))
    grp_agg['AssessmentMax'] = max(list(np.unique(grp['assessment'])))
    grp['pathology'] = grp['pathology'].map(
        {'BENIGN': 'benign', 'MALIGNANT': 'malignant', 'BENIGN_WITHOUT_CALLBACK': 'benign'})
    # print(grp['pathology'])
    # print(len(np.unique(grp['pathology'])))
    if len(np.unique(grp['pathology'])) > 1:
        if 'malignant' in list(np.unique(grp['pathology'])):
            grp_agg['Groundtruth'] = 'malignant'
    else:
        grp_agg['Groundtruth'] = np.unique(grp['pathology'])
    return grp_agg


# adding groundtruth to MIL mammogram csv
def add_caselevel_groundtruth_MIL_csvfile():
    df_modality = pd.read_csv(cbis_ddsm_trainingfile, sep=';')
    print(df_modality.shape)
    df_original_masstrain = pd.read_csv(cbis_ddsm_originalfile_masstrain, sep=',')
    df_original_masstrain['foldername'] = df_original_masstrain['image file path'].str.split('_').apply(
        lambda x: "_".join(x[:3]))
    # df_original.groupby(by=['foldername']).apply(lambda x: len(np.unique(x['pathology']))>1)
    df_patientfolder_masstrain = df_original_masstrain.groupby(by=['foldername'], as_index=False,
                                                               group_keys=False).apply(aggregating_patient_case_info)
    df_merged_masstrain = df_modality.merge(df_patientfolder_masstrain, on=['FolderName', 'PatientId'], how='inner')
    print(df_merged_masstrain.shape)
    print(df_merged_masstrain)
    input('halt')

    df_original_masstest = pd.read_csv(cbis_ddsm_originalfile_masstest, sep=',')
    df_original_masstest['foldername'] = df_original_masstest['image file path'].str.split('_').apply(
        lambda x: "_".join(x[:3]))
    df_patientfolder_masstest = df_original_masstest.groupby(by=['foldername'], as_index=False, group_keys=False).apply(
        aggregating_patient_case_info)
    df_merged_masstest = df_modality.merge(df_patientfolder_masstest, on=['FolderName', 'PatientId'], how='inner')
    print(df_merged_masstest.shape)
    print(df_merged_masstest)
    input('halt')

    df_original_calctrain = pd.read_csv(cbis_ddsm_originalfile_calctrain, sep=',')
    df_original_calctrain['foldername'] = df_original_calctrain['image file path'].str.split('_').apply(
        lambda x: "_".join(x[:3]))
    df_patientfolder_calctrain = df_original_calctrain.groupby(by=['foldername'], as_index=False,
                                                               group_keys=False).apply(aggregating_patient_case_info)
    df_merged_calctrain = df_modality.merge(df_patientfolder_calctrain, on=['FolderName', 'PatientId'], how='inner')
    print(df_merged_calctrain.shape)
    print(df_merged_calctrain)
    input('halt')

    df_original_calctest = pd.read_csv(cbis_ddsm_originalfile_calctest, sep=',')
    df_original_calctest['foldername'] = df_original_calctest['image file path'].str.split('_').apply(
        lambda x: "_".join(x[:3]))
    df_patientfolder_calctest = df_original_calctest.groupby(by=['foldername'], as_index=False, group_keys=False).apply(
        aggregating_patient_case_info)
    df_merged_calctest = df_modality.merge(df_patientfolder_calctest, on=['FolderName', 'PatientId'], how='inner')
    print(df_merged_calctest.shape)
    print(df_merged_calctest)

    df_merged = [df_merged_masstrain, df_merged_masstest, df_merged_calctrain, df_merged_calctest]
    df_merged = pd.concat(df_merged)
    df_merged = df_merged.sort_values(by='PatientId')
    print(df_merged)
    print(df_merged[~df_merged['Groundtruth'].isnull()].shape)

    df_merged.to_csv('/home/MG_training_files_cbis-ddsm_multiinstance_groundtruth.csv', sep=';')


if file_creation == 'MIL':
    # input_foldername = r"/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/clean_data"  # <path to cleaned/preprocessed images>
    output_foldername = r"/home/kemove/PycharmProjects/multiinstance-learning-mammography/datasets/cbis/mil_data"  # <path to multi-instance data>
    # dicom_folder = "C:/Users/PathakS/Shreyasi/CBIS-DDSM/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM"

    # ROI input files provided by cbis-ddsm website
    cbis_ddsm_trainingfile = r'/home/kemove/PycharmProjects/multiinstance-learning-mammography/input-csv-files/cbis/MG_training_files_vindr_multiinstance.csv'
    cbis_ddsm_originalfile_masstrain = r'/media/kemove/杨翻盖/data/ddsm/csv/mass_case_description_train_set.csv'
    cbis_ddsm_originalfile_masstest = r'/media/kemove/杨翻盖/data/ddsm/csv/mass_case_description_test_set.csv'
    cbis_ddsm_originalfile_calctrain = r'/media/kemove/杨翻盖/data/ddsm/csv/calc_case_description_train_set.csv'
    cbis_ddsm_originalfile_calctest = r'/media/kemove/杨翻盖/data/ddsm/csv/calc_case_description_test_set.csv'

    # creating_MIL_folder_structure(input_foldername, output_foldername)
    create_MIL_csv_file(cbis_ddsm_trainingfile)
    # add_caselevel_groundtruth_MIL_csvfile()
# ------------------------------------------------------------------MIL csv file creation end----------------------------------------------#
