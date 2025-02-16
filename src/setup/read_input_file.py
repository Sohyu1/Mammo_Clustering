import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utilities import utils


def dataset_based_changes(config_params):
    if config_params['dataset'] == 'cbis-ddsm':
        views_col = 'Views'
        sanity_check_mil_col = 'FolderName'
        sanity_check_sil_col = 'ImageName'

    elif config_params['dataset'] == 'vindr' or config_params['dataset'] == 'private':
        views_col = 'Views'
        sanity_check_mil_col = 'ShortPath'
        sanity_check_sil_col = 'CasePath'

    return views_col, sanity_check_mil_col, sanity_check_sil_col


def input_file_creation(config_params):
    views_col, sanity_check_mil_col, sanity_check_sil_col = dataset_based_changes(config_params)
    if config_params['learningtype'] == 'SIL':
        if config_params['datasplit'] == 'officialtestset':
            csv_file_path = config_params['SIL_csvfilepath']
            df_modality = pd.read_csv(csv_file_path)
            df_modality = df_modality[~df_modality['Groundtruth'].isnull()]
            df_modality['FullPath'] = config_params['preprocessed_imagepath'] + '/' + df_modality['path'] + '/' + \
                                      df_modality['Views'] + '.png'

            if config_params['dataset'] == 'cbis-ddsm':
                df_train = df_modality[df_modality['Split'].str.contains('training')]
                if config_params['usevalidation']:
                    df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True,
                                                        stratify=df_train['Groundtruth'])
                df_test = df_modality[df_modality['Split'].str.contains('test')]

            elif config_params['dataset'] == 'vindr':
                df_train = df_modality[df_modality['Split'] == 'training']
                if config_params['usevalidation']:
                    df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
                df_test = df_modality[df_modality['Split'] == 'test']
            total_instances = df_modality.shape[0]

    elif config_params['learningtype'] == 'MIL' or config_params['learningtype'] == 'MV':
        if config_params['datasplit'] == 'casebasedtestset':
            csv_file_path = config_params['MIL_csvfilepath']
            df_modality = pd.read_csv(csv_file_path, sep=';')
            df_modality = df_modality[~df_modality['Views'].isnull()]
            df_modality['FullPath'] = config_params['preprocessed_imagepath'] + '/' + df_modality['ShortPath']
            if config_params['numclasses'] == 3:
                df_modality['Groundtruth'] = df_modality['Groundtruth_3class']
            elif (config_params['numclasses'] == 1) or (config_params['numclasses'] == 2):
                try:
                    df_modality['Groundtruth'] = df_modality['Groundtruth_2class']
                except:
                    pass

            # bags with exactly 4 views
            df_modality1 = df_modality[df_modality[views_col].str.split('+').str.len() == 4.0]
            df_train, df_val, df_test = utils.stratifiedgroupsplit(df_modality1, config_params['randseeddata'])
            total_instances = df_modality1.shape[0]

            # bags with views!=4
            if config_params['viewsinclusion'] == 'all':
                df_modality2 = df_modality[df_modality[views_col].str.split('+').str.len() != 4.0]
                df_train = pd.concat(
                    [df_train, df_modality2[df_modality2['Patient_Id'].isin(df_train['Patient_Id'].unique().tolist())]])
                df_val = pd.concat(
                    [df_val, df_modality2[df_modality2['Patient_Id'].isin(df_val['Patient_Id'].unique().tolist())]])
                df_test = pd.concat(
                    [df_test, df_modality2[df_modality2['Patient_Id'].isin(df_test['Patient_Id'].unique().tolist())]])
                df_modality2 = df_modality2[~df_modality2['Patient_Id'].isin(
                    df_train['Patient_Id'].unique().tolist() + df_val['Patient_Id'].unique().tolist() + df_test[
                        'Patient_Id'].unique().tolist())]
                df_train1, df_val1, df_test1 = utils.stratifiedgroupsplit(df_modality2, config_params['randseeddata'])

                df_train = pd.concat([df_train, df_train1])
                df_val = pd.concat([df_val, df_val1])
                df_test = pd.concat([df_test, df_test1])
                total_instances = df_modality.shape[0]

        elif config_params['datasplit'] == 'officialtestset':
            csv_file_path = config_params['MIL_csvfilepath']
            df_modality = pd.read_csv(csv_file_path, sep=';')
            df_modality = df_modality[~df_modality['Views'].isnull()]

            # bags with all views
            if config_params['viewsinclusion'] == 'all':
                df_modality['FullPath'] = config_params['preprocessed_imagepath'] + '/' + df_modality['ShortPath']
                if config_params['dataset'] == 'vindr' or config_params['dataset'] == 'private':
                    df_train = df_modality[df_modality['Split'] == 'training']
                    if config_params['usevalidation']:
                        df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True,
                                                            stratify=df_train['Groundtruth'])
                    df_test = df_modality[df_modality['Split'] == 'test']
                    total_instances = df_modality.shape[0]

                elif config_params['dataset'] == 'cbis-ddsm':
                    df_train = df_modality[df_modality['FolderName'].str.contains('Training')]
                    if config_params['usevalidation']:
                        df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True,
                                                            stratify=df_train['Groundtruth'])
                    df_test = df_modality[df_modality['FolderName'].str.contains('Test')]
                    total_instances = df_modality.shape[0]

            elif config_params['viewsinclusion'] == 'standard':
                df_modality1 = df_modality[df_modality[views_col].str.split('+').str.len() == 4.0]
                df_train = df_modality1[df_modality1['Split'] == 'training']
                if config_params['usevalidation']:
                    df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True,
                                                        stratify=df_train['Groundtruth'])
                df_test = df_modality1[df_modality1['Split'] == 'test']
                total_instances = df_modality1.shape[0]

    elif config_params['learningtype'] == 'DS':
        csv_file_path = config_params['DS_csvfilepath']
        df_modality = pd.read_csv(csv_file_path, sep=';')
        df_modality = df_modality[~df_modality['Views'].isnull()]
        if config_params['dataset'] == 'vindr' or config_params['dataset'] == 'private':
            df_modality['FullPath'] = config_params['preprocessed_imagepath'] + '/' + df_modality['path']
        elif config_params['dataset'] == 'cbis-ddsm':
            df_modality['FullPath'] = config_params['preprocessed_imagepath'] + '/' + df_modality['ShortPath']

        # bags with all views
        if config_params['viewsinclusion'] == 'all':
            if config_params['dataset'] == 'vindr' or config_params['dataset'] == 'private':
                df_train = df_modality[df_modality['Split'] == 'training']
                if config_params['usevalidation']:
                    df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True,
                                                        stratify=df_train['Groundtruth'])
                df_test = df_modality[df_modality['Split'] == 'test']
                total_instances = df_modality.shape[0]

            elif config_params['dataset'] == 'cbis-ddsm':
                df_train = df_modality[df_modality['FolderName'].str.contains('Training')]
                if config_params['usevalidation']:
                    df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True,
                                                        stratify=df_train['Groundtruth'])
                df_test = df_modality[df_modality['FolderName'].str.contains('Test')]
                total_instances = df_modality.shape[0]

        elif config_params['viewsinclusion'] == 'standard':
            df_modality1 = df_modality[df_modality[views_col].str.split('+').str.len() == 4.0]
            df_train = df_modality1[df_modality1['Split'] == 'training']
            if config_params['usevalidation']:
                df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True,
                                                    stratify=df_train['Groundtruth'])
            df_test = df_modality1[df_modality1['Split'] == 'test']
            total_instances = df_modality1.shape[0]

    print("Total instances:", total_instances)

    df_train = df_train.reset_index()
    train_instances = df_train.shape[0]
    print("Train:", utils.stratified_class_count(df_train))
    print("training instances:", train_instances)
    if config_params['usevalidation']:
        df_val = df_val.reset_index()
        val_instances = df_val.shape[0]
        print("Val:", utils.stratified_class_count(df_val))
        print("Validation instances:", val_instances)
    df_test = df_test.reset_index()
    test_instances = df_test.shape[0]
    print("Test:", utils.stratified_class_count(df_test))
    print("Test instances:", test_instances)

    # if config_params['viewsinclusion'] == 'all' and config_params['learningtype'] == 'MIL':
    if config_params['viewsinclusion'] == 'all':
        # group by view
        view_group_indices, view_group_names_train = utils.groupby_view_train(df_train)

        df_val, view_group_names_val = utils.groupby_view_test(df_val)

        df_test, view_group_names_test = utils.groupby_view_test(df_test)
    else:
        view_group_indices = None

    # calculate number of batches
    # if config_params['viewsinclusion'] == 'all' and config_params['learningtype'] == 'MIL':
    if config_params['viewsinclusion'] == 'all':
        numbatches_train = int(
            sum(np.ceil(np.array(list(view_group_names_train.values())) / config_params['batchsize'])))
        numbatches_val = int(sum(np.ceil(np.array(list(view_group_names_val.values())) / config_params['batchsize'])))
        numbatches_test = int(sum(np.ceil(np.array(list(view_group_names_test.values())) / config_params['batchsize'])))
    else:
        numbatches_train = int(math.ceil(train_instances / config_params['batchsize']))

        if config_params['usevalidation']:
            numbatches_val = int(math.ceil(val_instances / config_params['batchsize']))

        numbatches_test = int(math.ceil(test_instances / config_params['batchsize']))

    if config_params['usevalidation']:
        return df_train, df_val, df_test, numbatches_train, numbatches_val, numbatches_test, view_group_indices
    else:
        return df_train, df_test, numbatches_train, numbatches_test, view_group_indices
