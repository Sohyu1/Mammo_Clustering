# -*- coding: utf-8 -*-
import os
import math
import torch
import argparse
import random
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics
from src.setup.config_file_creation import create_config
import torch.nn.functional as F

from src.train_eval import test, optimization, loss_function, evaluation, data_loader
from src.models import sil_mil_model, wu_resnet, dual_stream
from src.utilities import pytorchtools, utils, dynamic_training_utils
from src.setup import read_config_file, read_input_file, output_files_setup

pd.set_option('display.max_colwidth', None)  # or 199


def compare_model_weights(model1_weights, model2_weights):
    # 确保两个模型的参数数量相同
    if len(model1_weights) != len(model2_weights):
        print("模型参数数量不一致。")

    # 逐个对比参数
    for (key1, value1), (key2, value2) in zip(model1_weights.items(), model2_weights.items()):
        if key1 != key2:
            print(f"参数名称不一致: {key1} != {key2}")
        if not (value1 == value2).all():
            print(f"参数值不一致: {key1}")
        if (value1 == value2).all():
            print(f"参数值一致: {key1}")


def set_random_seed(config_params):
    # random state initialization of the code - values - 8, 24, 30
    torch.manual_seed(config_params['randseedother'])
    torch.cuda.manual_seed(config_params['randseedother'])
    torch.cuda.manual_seed_all(config_params['randseedother'])
    np.random.seed(config_params['randseeddata'])
    random.seed(config_params['randseeddata'])
    g = torch.Generator()
    g.manual_seed(config_params['randseedother'])
    torch.backends.cudnn.deterministic = True
    return g


def model_initialization(config_params):
    if config_params['learningtype'] == 'SIL':
        model = sil_mil_model.SILmodel(config_params)
    elif config_params['learningtype'] == 'MIL':
        model = sil_mil_model.MILmodel(config_params)
    elif config_params['learningtype'] == 'MV':
        model = wu_resnet.SplitBreastModel(config_params)
    elif config_params['learningtype'] == 'DS':
        model = dual_stream.DSmodel(config_params)
    if config_params['device'] == 'cuda':
        model = nn.DataParallel(model, device_ids=[0])
    model.to(torch.device(config_params['device']))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total model parameters:", pytorch_total_params, flush=True)

    return model, pytorch_total_params


def model_checkpoint(config_params, path_to_model):
    if config_params['patienceepochs']:
        modelcheckpoint = pytorchtools.EarlyStopping(path_to_model=path_to_model,
                                                     early_stopping_criteria=config_params['early_stopping_criteria'],
                                                     patience=config_params['patienceepochs'], verbose=True)
    elif config_params['usevalidation']:
        modelcheckpoint = pytorchtools.ModelCheckpoint_test(path_to_model=path_to_model, verbose=True)
    return modelcheckpoint


def train(config_params, model, path_to_model, data_iterator_train, data_iterator_val, batches_train, batches_val,
          df_train, dataloader_test, batches_test, df_test):
    """Training"""
    if config_params['usevalidation']:
        modelcheckpoint = model_checkpoint(config_params, path_to_model)
    optimizer = optimization.optimizer_fn(config_params, model)
    scheduler = optimization.select_lr_scheduler(config_params, optimizer)
    class_weights_train = loss_function.class_imbalance(config_params, df_train)

    if os.path.isfile(path_to_model):
        model, optimizer, start_epoch = utils.load_model(model, optimizer, path_to_model)
        print("start epoch:", start_epoch)
        print("lr:", optimizer.param_groups[0]['lr'])
    else:
        start_epoch = 0

    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_function.loss_fn_gmic_initialize(config_params, class_weights_train,
                                                                      test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn = loss_function.loss_fn_crossentropy(config_params, class_weights_train, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn = loss_function.loss_fn_bce(config_params, class_weights_train, test_bool=False)

    for epoch in range(start_epoch, config_params['maxepochs']):
        model.train()
        # params = copy.deepcopy(model.state_dict())
        loss_train = 0.0
        correct_train = 0
        conf_mat_train = np.zeros((config_params['numclasses'], config_params['numclasses']))
        total_images_train = 0
        batch_no = 0
        eval_mode = False

        if config_params['trainingmethod'] == 'multisteplr1':
            model = utils.layer_selection_for_training(model, epoch, config_params['trainingmethod'], epoch_step=5)


        for train_idx, train_batch, train_labels, views_names, studyuid_path in data_iterator_train:
        # for train_idx, train_batch, train_labels, views_names in data_iterator_train:
            train_batch = train_batch.to(config_params['device'])
            train_labels = train_labels.to(config_params['device'])
            train_labels = train_labels.view(-1)

            if config_params['viewsinclusion'] == 'all' and config_params['extra'] == 'dynamic_training':
                model, optimizer, state_before_optim, lr_before_optim = dynamic_training_utils.dynamic_training(
                    config_params, views_names, model, optimizer, None, None, True)

            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _ = model(
                        train_batch, eval_mode)  # compute model output, loss and total train loss over one epoch
                    output_patch = None
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _, output_patch = model(
                        train_batch, views_names, eval_mode)

                if config_params['activation'] == 'sigmoid':
                    output_batch_local = output_batch_local.view(-1)
                    output_batch_global = output_batch_global.view(-1)
                    output_batch_fusion = output_batch_fusion.view(-1)
                    train_labels = train_labels.float()
                    pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()  # 实现大于等于（≥ \ge≥）运算

                elif config_params['activation'] == 'softmax':
                    pred = output_batch_fusion.argmax(dim=1, keepdim=True)
                loss = loss_function.loss_fn_gmic(config_params, bcelogitloss, bceloss, output_batch_local,
                                                  output_batch_global, output_batch_fusion, saliency_map, train_labels,
                                                  class_weights_train, output_patch, test_bool=False)

            else:
                if config_params['learningtype'] == 'SIL':
                    output_batch = model(train_batch, eval_mode)
                elif config_params['learningtype'] == 'MIL':
                    output_batch = model(train_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'MV':
                    output_batch = model(train_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'DS':
                    output_batch = model(train_batch, eval_mode)
                if config_params['activation'] == 'sigmoid':
                    if len(output_batch.shape) > 1:
                        output_batch = output_batch.squeeze(1)
                    output_batch = output_batch.view(-1)
                    train_labels = train_labels.float()
                    pred = torch.ge(torch.sigmoid(output_batch), torch.tensor(0.5)).float()
                    loss = lossfn(output_batch, train_labels)

                elif config_params['activation'] == 'softmax':
                    pred = output_batch.argmax(dim=1, keepdim=True)
                    loss = lossfn(output_batch, train_labels)

            loss_train += (train_labels.size()[0] * loss.item())
            optimizer.zero_grad()  # clear previous gradients, compute gradients of all variables wrt loss
            loss.backward()
            optimizer.step()  # performs updates using calculated gradients
            batch_no = batch_no + 1
            # params1 = model.state_dict()
            # compare_model_weights(params, params1)

            if config_params['viewsinclusion'] == 'all' and config_params['extra'] == 'dynamic_training':
                model, optimizer = dynamic_training_utils.dynamic_training(config_params, views_names, model, optimizer,
                                                                           state_before_optim, lr_before_optim, False)

            # performance metrics of training dataset  训练数据集的性能指标
            correct_train, total_images_train, conf_mat_train, _ = evaluation.conf_mat_create(pred, train_labels,
                                                                                              correct_train,
                                                                                              total_images_train,
                                                                                              conf_mat_train,
                                                                                              config_params['classes'])
            print('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, config_params['maxepochs'],
                                                                            batch_no, batches_train, loss.item()),
                  flush=True)
        #
        if scheduler != None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        running_train_loss = loss_train / total_images_train

        if config_params['usevalidation']:
            correct_val, total_images_val, loss_val, conf_mat_val, auc_val = validation(config_params, model,
                                                                                        data_iterator_val, batches_val,
                                                                                        df_val, epoch)

            valid_loss = loss_val / total_images_val
            evaluation.results_store_excel(True, True, False, None, correct_train, total_images_train, loss_train,
                                           correct_val, total_images_val, loss_val, epoch, conf_mat_train,
                                           conf_mat_val, current_lr, auc_val, path_to_results_xlsx,
                                           path_to_results_text)

        if config_params['usevalidation']:
            per_model_metrics, conf_mat_test = test.run_test_every(config_params, model, dataloader_test,
                                                                   batches_test, df_test,
                                                                   path_to_results_xlsx,
                                                                   'test_results', epoch)
            test_loss, test_f1, test_auc = per_model_metrics[0], per_model_metrics[9], per_model_metrics[13]
            modelcheckpoint(test_loss, model, optimizer, epoch, conf_mat_train, conf_mat_val, running_train_loss,
                            test_auc)
        else:
            utils.save_model(model, optimizer, epoch, running_train_loss, path_to_model)
            per_model_metrics, conf_mat_test, best_epoch, best_auc = test.run_test_every(config_params, model,
                                                                                         dataloader_test,
                                                                                         batches_test, df_test,
                                                                                         path_to_results_xlsx,
                                                                                         'test_results', epoch,
                                                                                         best_epoch, best_auc)
            evaluation.results_store_excel(True, False, True, per_model_metrics, correct_train, total_images_train,
                                           loss_train, None, None, None, epoch, conf_mat_train, None, current_lr,
                                           None, path_to_results_xlsx, path_to_results_text)
            evaluation.write_results_xlsx_confmat(config_params, conf_mat_test, path_to_results_xlsx,
                                                  'confmat_train_val_test')
            evaluation.write_results_xlsx(per_model_metrics, path_to_results_xlsx, 'test_results')

        if scheduler != None:
            scheduler.step()

    if config_params['usevalidation']:
        evaluation.write_results_xlsx_confmat(config_params, modelcheckpoint.conf_mat_train_best, path_to_results_xlsx,
                                              'confmat_train_val_test')

    print('Finished Training')


def validation(config_params, model, data_iterator_val, batches_val, df_val, epoch):
    """Validation"""
    model.eval()
    total_images = 0
    val_loss = 0
    correct = 0
    s = 0
    batch_val_no = 0
    eval_mode = True
    conf_mat_val = np.zeros((config_params['numclasses'], config_params['numclasses']))

    class_weights_val = loss_function.class_imbalance(config_params, df_val)

    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss_val, bceloss_val = loss_function.loss_fn_gmic_initialize(config_params, class_weights_val,
                                                                              test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_function.loss_fn_crossentropy(config_params, class_weights_val, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_function.loss_fn_bce(config_params, class_weights_val, test_bool=False)

    with torch.no_grad():
        for val_idx, val_batch, val_labels, views_names, studyuid_path in data_iterator_val:
        # for val_idx, val_batch, val_labels, views_names in data_iterator_val:
            val_batch, val_labels = val_batch.to(config_params['device']), val_labels.to(config_params['device'])
            val_labels = val_labels.view(-1)  # .float()
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, _, _, _, _ = model(
                        val_batch, eval_mode)  # compute model output, loss and total train loss over one epoch
                    output_patch_val = None
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, _, _, _, _, output_patch_val = model(
                        val_batch, views_names, eval_mode)

                if config_params['activation'] == 'sigmoid':
                    output_batch_local_val = output_batch_local_val.view(-1)
                    output_batch_global_val = output_batch_global_val.view(-1)
                    output_batch_fusion_val = output_batch_fusion_val.view(-1)
                    val_labels = val_labels.float()
                    val_pred = torch.ge(torch.sigmoid(output_batch_fusion_val), torch.tensor(0.5)).float()

                elif config_params['activation'] == 'softmax':
                    val_pred = output_batch_fusion_val.argmax(dim=1, keepdim=True)

                loss1 = loss_function.loss_fn_gmic(config_params, bcelogitloss_val, bceloss_val, output_batch_local_val,
                                                   output_batch_global_val, output_batch_fusion_val, saliency_map_val,
                                                   val_labels, class_weights_val, output_patch_val,
                                                   test_bool=False).item()
                output_val = output_batch_fusion_val
            else:
                if config_params['learningtype'] == 'SIL':
                    output_val = model(val_batch, eval_mode)
                elif config_params['learningtype'] == 'MIL':
                    output_val = model(val_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'MV':
                    output_val = model(val_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'DS':
                    output_val = model(val_batch, eval_mode)
                if config_params['activation'] == 'sigmoid':
                    if len(output_val.shape) > 1:
                        output_val = output_val.squeeze(1)
                    output_val = output_val.view(-1)
                    val_labels = val_labels.float()
                    val_pred = torch.ge(torch.sigmoid(output_val), torch.tensor(0.5)).float()
                    loss1 = lossfn1(output_val, val_labels).item()
                elif config_params['activation'] == 'softmax':
                    val_pred = output_val.argmax(dim=1, keepdim=True)
                    loss1 = lossfn1(output_val, val_labels).item()

            if batch_val_no == 0:
                val_pred_all = val_pred
                val_labels_all = val_labels
                print(output_val.data.shape, flush=True)
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.sigmoid(output_val.data)
                elif config_params['activation'] == 'softmax':
                    output_all_ten = F.softmax(output_val.data, dim=1)
                    if config_params['numclasses'] < 3:
                        output_all_ten = output_all_ten[:, 1]
            else:
                val_pred_all = torch.cat((val_pred_all, val_pred), dim=0)
                val_labels_all = torch.cat((val_labels_all, val_labels), dim=0)
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.cat((output_all_ten, torch.sigmoid(output_val.data)), dim=0)
                elif config_params['activation'] == 'softmax':
                    if config_params['numclasses'] < 3:
                        output_all_ten = torch.cat((output_all_ten, F.softmax(output_val.data, dim=1)[:, 1]), dim=0)
                    else:
                        output_all_ten = torch.cat((output_all_ten, F.softmax(output_val.data, dim=1)), dim=0)

            s = s + val_labels.shape[0]
            val_loss += val_labels.size()[0] * loss1  # sum up batch loss
            correct, total_images, conf_mat_val, _ = evaluation.conf_mat_create(val_pred, val_labels, correct,
                                                                                total_images, conf_mat_val,
                                                                                config_params['classes'])

            batch_val_no += 1
            print('Val: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, config_params['maxepochs'],
                                                                          batch_val_no, batches_val, loss1), flush=True)

    print("conf_mat_val:", conf_mat_val, flush=True)
    print("total_images:", total_images, flush=True)
    print("s:", s, flush=True)
    print('\nVal set: total val loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch:{}\n'.format(
        val_loss, val_loss / total_images, correct, total_images,
                  100. * correct / total_images, epoch + 1), flush=True)

    auc = metrics.roc_auc_score(val_labels_all.cpu().numpy(), output_all_ten.cpu().numpy(), multi_class='ovo')
    return correct, total_images, val_loss, conf_mat_val, auc


if __name__ == '__main__':
    # read arguments
    create_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        default=r'D:\Code\Python_Code\Mammo\src\out_res\run\config_80_8.ini',
        help="full path where the config.ini file containing the parameters to run this code is stored",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="model training or test",
        default='train',  # test
    )
    args = parser.parse_args()

    mode = args.mode

    config_file = args.config_file_path
    config_params = read_config_file.read_config_file(config_file)
    config_params['path_to_output'] = "/".join(config_file.split('/')[:-1])
    g = set_random_seed(config_params)

    path_to_model, path_to_results_xlsx, path_to_results_text, path_to_learning_curve, path_to_log_file = output_files_setup.output_files(
        config_params)
    df_train, df_val, df_test, batches_train, batches_val, batches_test, view_group_indices_train = read_input_file.input_file_creation(
        config_params)
    dataloader_train, dataloader_val, dataloader_test = data_loader.dataloader(config_params, df_train, df_val,
                                                                               df_test,
                                                                               view_group_indices_train, g)

    model, total_params = model_initialization(config_params)

    if mode == 'train':
        train(config_params, model, path_to_model, dataloader_train, dataloader_val, batches_train, batches_val,
              df_train, dataloader_test, batches_test, df_test)
    else:
        test.run_test(config_params, model, path_to_model, dataloader_test, batches_test, df_test,
                      path_to_results_xlsx,
                      'test_results', 0)

    f = open(path_to_log_file, 'a')
    f.write("Model parameters:" + str(total_params / math.pow(10, 6)) + '\n')
    f.close()
