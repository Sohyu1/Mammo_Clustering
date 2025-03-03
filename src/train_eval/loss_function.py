import torch
import torch.nn as nn
from torch import linalg as LA
import torch.nn.functional as F

from src.utilities import utils


def class_imbalance(config_params, df):
    if config_params['classimbalance'] == 'poswt':
        class_weights = utils.class_distribution_poswt(df).to(config_params['device'])
    elif config_params['classimbalance'] == 'wtcostfunc':
        class_weights = utils.class_distribution_weightedloss(config_params, df).to(config_params['device'])
    else:
        class_weights = None
    return class_weights


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def loss_fn_crossentropy(config_params, class_weights, test_bool):
    if test_bool:
        criterion = nn.CrossEntropyLoss()
    else:
        if config_params['classimbalance'] == 'poswt':
            class_weights = torch.tensor([1, class_weights[0]]).float().to(config_params['device'])
            if config_params['extra'] == 'labelsmoothing':
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            else:
                criterion = nn.CrossEntropyLoss(class_weights)
        elif config_params['classimbalance'] == 'wtcostfunc':
            criterion = nn.CrossEntropyLoss(class_weights)
        else:
            if config_params['extra'] == 'labelsmoothing':
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            else:
                criterion = nn.CrossEntropyLoss()
    return criterion


def loss_fn_bce(config_params, class_weights, test_bool):
    if test_bool or not config_params['classimbalance']:
        criterion = nn.BCEWithLogitsLoss()

    elif config_params['classimbalance'] == 'poswt':
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[0])

    return criterion


def loss_fn_gmic_initialize(config_params, class_weights, test_bool):
    if config_params['activation'] == 'sigmoid':
        if test_bool or not config_params['classimbalance']:
            logitloss = nn.BCEWithLogitsLoss()
            loss = nn.BCELoss()

        elif config_params['classimbalance'] == 'poswt':
            logitloss = nn.BCEWithLogitsLoss(pos_weight=class_weights[0])
            loss = nn.BCELoss()

    elif config_params['activation'] == 'softmax':
        if test_bool or not config_params['classimbalance']:
            logitloss = nn.CrossEntropyLoss()
            loss = nn.CrossEntropyLoss()

        elif config_params['classimbalance'] == 'wtcostfunc':
            logitloss = nn.CrossEntropyLoss(class_weights)
            loss = nn.NLLLoss(class_weights)

    return logitloss, loss


def loss_fn_gmic(config_params, logitloss, loss, y_local, y_global, y_fusion, saliency_map, y_true, class_weights,
                 y_patch, test_bool):
    if config_params['classimbalance'] == 'poswt' and not test_bool:
        weight_batch = torch.tensor([1, class_weights[0]]).to(config_params['device'])[y_true.long()]
        loss.weight = weight_batch
    local_network_loss = logitloss(y_local, y_true)
    if config_params['activation'] == 'sigmoid':
        y_global = torch.clamp(y_global, 0, 1)
        global_network_loss = loss(y_global, y_true)

    elif config_params['activation'] == 'softmax':
        global_network_loss = logitloss(y_global, y_true)
    fusion_network_loss = logitloss(y_fusion, y_true)
    if config_params['learningtype'] == 'SIL':
        saliency_map_regularizer = torch.mean(
            LA.norm(saliency_map.view(saliency_map.shape[0], saliency_map.shape[1], -1), ord=1, dim=2))
    elif config_params['learningtype'] == 'MIL':
        if config_params['numclasses'] == 3:
            # print("Atten CNN:", saliency_map.shape) #N,4,3,92,60
            saliency_map_regularizer = torch.mean(torch.mean(
                LA.norm(saliency_map.view(saliency_map.shape[0], saliency_map.shape[1], saliency_map.shape[2], -1),
                        ord=1, dim=3), dim=1))
        else:
            saliency_map_regularizer = torch.mean(
                torch.mean(LA.norm(saliency_map.view(saliency_map.shape[0], saliency_map.shape[1], -1), ord=1, dim=2),
                           dim=1))

    total_loss = local_network_loss + global_network_loss + fusion_network_loss + config_params[
        'sm_reg_param'] * saliency_map_regularizer  # + patch_loss
    return total_loss


def loss_fn_ds(config_params, loss, y_local, y_global, y_fusion, y_true):
    local_network_loss = loss(y_local, y_true)
    if config_params['activation'] == 'sigmoid':
        y_global = torch.clamp(y_global, 0, 1)
        global_network_loss = loss(y_global, y_true)
    elif config_params['activation'] == 'softmax':
        global_network_loss = loss(y_global, y_true)
    fusion_network_loss = loss(y_fusion, y_true)

    total_loss = 0.5 * (local_network_loss + global_network_loss) + 0.7 * fusion_network_loss  # + patch_loss
    return total_loss
