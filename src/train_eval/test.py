# -*- coding: utf-8 -*-
import csv
import json
import os
from copy import deepcopy
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from src.train_eval import loss_function, evaluation
from src.utilities import utils


def visualization(test_idx, test_batch, patch_locations, patches, studyuid_path):
    views = ['LCC', 'LMLO', 'RCC',  'RMLO']
    for q in range(len(test_idx)):
        for i in range(len(test_batch[0])):
            test_batch_int = (test_batch[q, i, 0, :, :] * 22).int().cpu().numpy()
            # plt.imshow(test_batch_int, cmap='gray')
            # plt.show()
            h, w = test_batch_int.shape[0], test_batch_int.shape[1]
            path_stu = studyuid_path[q].split('/')[-1]

            for j in range(len(patch_locations[q, i])):
                # patches_int = (patches[q, i, j, :, :] * 22).int().cpu().numpy()
                y1 = int(patch_locations[q, i, j, 0])
                y2 = int(patch_locations[q, i, j, 0]) + 224
                x1 = int(patch_locations[q, i, j, 1])
                x2 = int(patch_locations[q, i, j, 1]) + 224
                cv2.rectangle(test_batch_int, (x1, y1),
                              (x2, y2),
                              (0, 255), 15)
                if views[i] == 'RMLO' or views[i] == 'RCC':
                    x2 = w - x1
                    x1 = w - x1 - 224
                with open(r"D:\Code\Python_Code\Mammo\experiment\detection\gmic\cal_model_res34.csv",
                          "a+", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([studyuid_path[q].split('/')[-1], views[i], x1, y1, x2, y2])
            #     cv2.putText(test_batch_int, str(j), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200), 5)
            #     plt.imshow(test_batch_int, cmap='gray')  # 可视化
            #     plt.show()
            #     plt.imshow(patches_int, cmap='gray')  # 可视化
            #     plt.show()
            #     patch_path = r'D:\Code\Python_Code\Mammo\datasets\img_with_patch/{}/patch'.format(path_stu)
            #     if not os.path.exists(patch_path):  # 如果路径不存在
            #         os.makedirs(patch_path)
            #     plt.imsave(patch_path + '/{}_patch_{}.png'.format(views[i], j), patches_int, cmap='gray')
            # path = r'D:\Code\Python_Code\Mammo\experiment\visualized_images\coc\vis_patch/{}'.format(path_stu)
            # if not os.path.exists(path):  # 如果路径不存在
            #     os.makedirs(path)
            # plt.imsave(path + '/{}.png'.format(views[i]), test_batch_int, cmap='gray')


def load_model_for_testing(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    print("checkpoint epoch and loss:", checkpoint['epoch'], checkpoint['loss'])
    return model


def test_function(config_params, model, dataloader_test, batches_test, df_test, path_to_results_xlsx, sheetname, epoch):
    """Testing"""
    model.eval()
    total_images = 0
    test_loss = 0
    correct = 0
    s = 0
    batch_test_no = 0
    count_dic_viewwise = {}
    eval_subgroup = True
    eval_mode = True
    conf_mat_test = np.zeros((config_params['numclasses'], config_params['numclasses']))
    views_standard = ['LCC', 'LMLO', 'RCC', 'RMLO']

    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_function.loss_fn_gmic_initialize(config_params, None, test_bool=True)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_function.loss_fn_crossentropy(config_params, None, test_bool=True)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_function.loss_fn_bce(config_params, None, test_bool=True)

    with torch.no_grad():
        with open(r"D:\Code\Python_Code\Mammo\experiment\detection\gmic\cal_model_res34.csv", "w",  newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["path", "view", "x1", "y1", "x2", "y2"])
        # y_trues = []
        # y_preds = []
        for test_idx, test_batch, test_labels, views_names, studyuid_path in dataloader_test:
        # for test_idx, test_batch, test_labels, views_names in dataloader_test:
            test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
            test_labels = test_labels.view(-1)
            if config_params['femodel'] == 'gmic_resnet18':
                output_batch_local, output_batch_global, output_batch_fusion, saliency_map, patch_locations, patches, _, _, output_patch_test = model(
                    test_batch, views_names, eval_mode)

                # # 可视化
                visualization(test_idx, test_batch, patch_locations, patches, studyuid_path)

                if config_params['activation'] == 'sigmoid':
                    output_batch_local = output_batch_local.view(-1)
                    output_batch_global = output_batch_global.view(-1)
                    output_batch_fusion = output_batch_fusion.view(-1)
                    test_labels = test_labels.float()
                    test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
                elif config_params['activation'] == 'softmax':
                    test_pred = output_batch_fusion.argmax(dim=1, keepdim=True)

                # for i in range(len(test_labels)):
                #     y_preds.append(torch.sigmoid(output_batch_fusion[i]).cpu())
                #     y_trues.append(test_labels[i].cpu())

                loss1 = loss_function.loss_fn_gmic(config_params, bcelogitloss, bceloss, output_batch_local,
                                                   output_batch_global, output_batch_fusion, saliency_map, test_labels,
                                                   None, output_patch_test, test_bool=True).item()
                output_test = output_batch_fusion

            else:
                if config_params['learningtype'] == 'SIL':
                    output_test = model(test_batch, eval_mode)
                elif config_params['learningtype'] == 'MIL':
                    output_test = model(test_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'MV':
                    output_test = model(test_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'DS':
                    output_test = model(test_batch, eval_mode)
                if config_params['activation'] == 'sigmoid':
                    if len(output_test.shape) > 1:
                        output_test = output_test.squeeze(1)
                    output_test = output_test.view(-1)
                    test_labels = test_labels.float()
                    test_pred = torch.ge(torch.sigmoid(output_test), torch.tensor(0.5)).float()
                    # for i in range(len(test_labels)):
                    #     y_preds.append(torch.sigmoid(output_test[i]).cpu())
                    #     y_trues.append(test_labels[i].cpu())
                    loss1 = lossfn1(output_test, test_labels).item()
                elif config_params['activation'] == 'softmax':
                    test_pred = output_test.argmax(dim=1, keepdim=True)
                    loss1 = lossfn1(output_test, test_labels).item()

            if batch_test_no == 0:
                test_pred_all = test_pred
                test_labels_all = test_labels
                loss_all = torch.tensor([loss1])
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.sigmoid(output_test.data)
                elif config_params['activation'] == 'softmax':
                    output_all_ten = F.softmax(output_test.data, dim=1)
                    if config_params['numclasses'] < 3:
                        output_all_ten = output_all_ten[:, 1]
            else:
                test_pred_all = torch.cat((test_pred_all, test_pred), dim=0)
                test_labels_all = torch.cat((test_labels_all, test_labels), dim=0)
                loss_all = torch.cat((loss_all, torch.tensor([loss1])), dim=0)
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.cat((output_all_ten, torch.sigmoid(output_test.data)), dim=0)
                elif config_params['activation'] == 'softmax':
                    if config_params['numclasses'] < 3:
                        output_all_ten = torch.cat((output_all_ten, F.softmax(output_test.data, dim=1)[:, 1]), dim=0)
                    else:
                        output_all_ten = torch.cat((output_all_ten, F.softmax(output_test.data, dim=1)), dim=0)

            test_loss += test_labels.size()[0] * loss1  # sum up batch loss
            correct, total_images, conf_mat_test, conf_mat_batch = evaluation.conf_mat_create(test_pred, test_labels,
                                                                                              correct, total_images,
                                                                                              conf_mat_test,
                                                                                              config_params['classes'])

            if config_params['viewsinclusion'] == 'all' and config_params['learningtype'] == 'MIL' and (
                    config_params['dataset'] == 'zgt' or config_params['dataset'] == 'cbis-ddsm'):
                count_dic_viewwise = evaluation.calc_viewwise_metric(views_names, views_standard, count_dic_viewwise,
                                                                     test_labels, test_pred, output_test)

            batch_test_no += 1
            s = s + test_labels.shape[0]
            print('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1), flush=True)

        # y_true1 = np.array(y_trues)
        # y_pred1 = np.array(y_preds)
        # # 计算ROC曲线的假正率（FPR）和真正率（TPR）
        # fpr, tpr, thresholds = roc_curve(y_true1, y_pred1)
        # with open(r'/experiment/ROC/wtconv/vindr\gmic.txt', 'w') as f:
        #     # 将列表转换为字符串，并用逗号分隔
        #     list1_str = ','.join(CNN(str, fpr))
        #     list2_str = ','.join(CNN(str, tpr))
        #
        #     # 将两个列表用分号分隔并写入文件
        #     f.write(f"{list1_str};{list2_str}")

    running_loss = test_loss / total_images
    print("conf_mat_test:", conf_mat_test, flush=True)
    print("total_images:", total_images, flush=True)
    print("s:", s, flush=True)
    print('\nTest set: total test loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n'.format(
        test_loss, running_loss, correct, total_images, 100. * correct / total_images), flush=True)

    if config_params['learningtype'] == 'SIL':
        per_model_metrics = evaluation.case_label_from_SIL(config_params, df_test, test_labels_all.cpu().numpy(),
                                                           test_pred_all.cpu().numpy(), path_to_results_xlsx, epoch)
        per_model_metrics = [running_loss] + per_model_metrics
    else:
        per_model_metrics = evaluation.aggregate_performance_metrics(config_params, test_labels_all.cpu().numpy(),
                                                                     test_pred_all.cpu().numpy(),
                                                                     output_all_ten.cpu().numpy(), epoch)
        per_model_metrics = [running_loss] + per_model_metrics

        if sheetname == 'hyperparam_results':
            hyperparam_details = [config_params['config_file'], config_params['lr'], config_params['wtdecay'],
                                  config_params['sm_reg_param'], config_params['trainingmethod'],
                                  config_params['optimizer'], config_params['patienceepochs'],
                                  config_params['batchsize']] + per_model_metrics
            evaluation.write_results_xlsx(hyperparam_details, config_params['path_to_hyperparam_search'],
                                          'hyperparam_results')
        else:
            evaluation.write_results_xlsx_confmat(config_params, conf_mat_test, path_to_results_xlsx,
                                                  'confmat_train_val_test')
            evaluation.write_results_xlsx(per_model_metrics, path_to_results_xlsx, 'test_results')
            evaluation.classspecific_performance_metrics(config_params, test_labels_all.cpu().numpy(),
                                                         test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy(),
                                                         path_to_results_xlsx, 'test_results')

    # changed
    return per_model_metrics, conf_mat_test


def run_test_every(config_params, model, dataloader_test, batches_test, df_test, path_to_results_xlsx,
             sheetname, epoch):
    # changed
    per_model_metrics, conf_mat_test = test_function(config_params, model, dataloader_test, batches_test, df_test, path_to_results_xlsx, sheetname, epoch)

    return per_model_metrics, conf_mat_test


def run_test(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_results_xlsx,
                   sheetname, epoch):
    path_to_trained_model = path_to_model
    model = load_model_for_testing(model, path_to_trained_model)
    per_model_metrics, conf_mat_test = test_function(config_params, model, dataloader_test, batches_test, df_test,
                                                     path_to_results_xlsx, sheetname, epoch)

