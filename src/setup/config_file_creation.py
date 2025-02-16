# -*- coding: utf-8 -*-
from configparser import ConfigParser
import os

hyperparam_config = [
    {'lr': 0.000006830957344480193, 'wtdecay': 0.000316227766016838, 'sm_reg_param': 0.000158489319246111}]

names = []
start = 0
end = len(hyperparam_config)
count = 0


def create_config():
    hyperparam = hyperparam_config[0]
    # Get the configparser object
    config_object = ConfigParser()
    config_object["parametersetting"] = {
        "modelid": 54,  # 54
        "run": False,
        "attention": 'imagewise',  # options = imagewise, breastwise, False
        "dependency": False,
        "selfatt-nonlinear": False,
        "selfatt-gamma": False,
        "milpooling": 'esatt',  # esatt
        # options=maxpool, average, attention, gatedattention, concat/ ismax, ismean, isatt, isgatt, esmax, esmean, esatt, esgatt
        "activation": 'sigmoid',  # options = sigmoid, softmax
        "viewsinclusion": 'all',
        # option = standard, all -> change this to viewsinclusion: standard, all; in SIL: standard means all views. I put standard to prevent the dynamic training part of the code.
        "classimbalance": 'poswt',  # options = wtcostfunc, poswt, oversampling, focalloss,False
        "optimizer": 'Adam',  # options = SGD, Adam
        "patienceepochs": 0,  # 10
        "usevalidation": True,
        "batchsize": 2,  # options=10, 20
        "numclasses": 1,
        "maxepochs": 150,  # 150
        "numworkers": 4,
        "lr": float(hyperparam['lr']),  # 10**float(hyperparam['lr']), #0.001, 0.00002
        "wtdecay": float(hyperparam['wtdecay']),  # 10**float(hyperparam['wtdecay']), #0.0005, 0.00001
        "sm_reg_param": float(hyperparam['sm_reg_param']),  # 10**float(hyperparam['sm_reg_param']), False
        "groundtruthdic": {'benign': 0, 'malignant': 1},  # {'normal':0,'benign':1,'malignant':2},
        "classes": [0, 1],  # [0,1,2],
        "resize": [2688, 896],  # options=1600, zgt, cbis-ddsm: [2944,1920], vindr:[2700, 990], None (for padding to max image size ) [2688, 896]
        "cam_size": (84, 28),  # vindr: (85, 31), zgt:(92, 60)  (84, 28)
        "crop_shape": (224, 224),  # (256, 256)
        "dataaug": 'gmic',  # options=small, big, wang, gmic, kim, shu
        "imagecleaning": 'own',
        "datasplit": 'officialtestset',  # options: officialtestset, casebasedtestset
        "datascaling": 'scaling',  # options=scaling, standardize, standardizeperimage,False
        "flipimage": True,
        "randseedother": 80,  # options=8, 24, 80
        "randseeddata": 8,  # options=8, 24, 80, 42  1:8; 2:24
        "device": 'cuda:0',
        "trainingmethod": 'StepLR',  # options: multisteplr1, fixedlr, lrdecayshu, lrdecaykim, cosineannealing, StepLR
        "channel": 3,  # options: 3 for rgb, 1 for grayscale
        "regionpooling": 'maxpool',  # options: shu_ggp, shu_rgp, avgpool, maxpool, 1x1conv, t-pool
        "femodel": 'gmic_resnet18',  # options: resnet50pretrainedrgbwang, densenet169, gmic_resnet18 sparsevit dual_steam
        "pretrained": True,  # options: True, False
        "topkpatch": 0.02,  # options: 0.02, 0.03, 0.05, 0.1
        "ROIpatches": 4,  # options: any number, 6 from gmic paper
        "learningtype": 'MIL',  # options = SIL, MIL, MV (multiview) DS(dual_stream)
        "dataset": 'cbis-ddsm',  # options = cbis-ddsm, vindr, private
        "bitdepth": 16,  # options: 8, 16
        "labeltouse": 'caselabel',  # options: imagelabel, caselabel

        "SIL_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\cbis\cbis_ddsm_sil.csv",
        # "SIL_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\vindr\vindr_sil.csv",

        # "MIL_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\private\private_mil.csv",
        "MIL_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\cbis\cbis_ddsm_mil.csv",
        # "MIL_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\vindr\vindr_mil.csv",

        # "DS_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\private\private_ds.csv",
        "DS_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\cbis\cbis_ddsm_ds.csv",
        # "DS_csvfilepath": r"D:\Code\Python_Code\Mammo\datasets\input-csv-files\vindr\vindr_ds.csv",

        # "preprocessed_imagepath": r'E:\data\private\clean_data',
        "preprocessed_imagepath": r'E:\data\cbis-ddsm\clean_data',
        # "preprocessed_imagepath": r'E:\data\vindr\clean_data',
        "valloss_resumetrain": False,
        "papertoreproduce": False,
        "early_stopping_criteria": 'loss',
        "extra": False  # options: dynamic_training
    }
    filename = ''

    for key in config_object["parametersetting"].keys():
        if key in ['modelid', 'attention', 'dependency', 'milpooling', 'femodel', 'viewsinclusion', 'papertoreproduce',
                   'learningtype', 'extra']:
            if config_object["parametersetting"][key] != 'False':
                if filename == '':
                    filename = key + config_object["parametersetting"][key]
                else:
                    filename = filename + '_' + key + config_object["parametersetting"][key]

    config_object["parametersetting"]['filename'] = filename
    path_to_output1 = r"D:/Code/Python_Code/Mammo/src/out_res/run"
    if config_object["parametersetting"]['dataset'] == 'cbis-ddsm':
        path_to_output = "D:/Code/Python_Code/Mammo/config/config_cbis"
    elif config_object["parametersetting"]['dataset'] == 'vindr':
        path_to_output = "D:/Code/Python_Code/Mammo/config/config_vindr"
    elif config_object["parametersetting"]['dataset'] == 'private':
        path_to_output = r"D:/Code/Python_Code/Mammo/config/config_private"

    # create output_folder path
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    if not os.path.exists(path_to_output1):
        os.mkdir(path_to_output1)

    if str(config_object["parametersetting"]["randseeddata"]) != str(
            config_object["parametersetting"]["randseedother"]):
        rand_seed = str(config_object["parametersetting"]["randseedother"]) + '_' + str(
            config_object["parametersetting"]["randseeddata"])
    else:
        rand_seed = str(config_object["parametersetting"]["randseeddata"])

    # Write the above sections to config.ini file
    if str(config_object["parametersetting"]["run"]) != 'False':
        with open(path_to_output + '/config_' + rand_seed + '_' + 'run_' + str(
                config_object["parametersetting"]["run"]) + '.ini', 'w') as conf:
            config_object.write(conf)
        with open(path_to_output1 + '/config_' + rand_seed + '_' + 'run_' + str(
                config_object["parametersetting"]["run"]) + '.ini', 'w') as conf:
            config_object.write(conf)
    else:
        with open(path_to_output + '/config_' + rand_seed + '.ini', 'w') as conf:
            config_object.write(conf)
        with open(path_to_output1 + '/config_' + rand_seed + '.ini', 'w') as conf:
            config_object.write(conf)


if __name__ == '__main__':
    create_config()
