
import math
import time

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from torch.autograd import Variable
from torchvision._internally_replaced_utils import load_state_dict_from_url

from src.utilities import data_augmentation_utils, gmic_utils
from src.models import gmic_modules as m

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, d_model, max_len=1.0):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        # x is expected to be of shape (batch_size, 2)
        num, batch_size, _ = x.size()

        # Create a tensor to hold the positional encodings
        pe = torch.zeros(num, batch_size, self.d_model)

        # Calculate the positional encodings
        for i in range(num):
            for pos in range(self.d_model // 2):
                div_term = math.exp(pos * -math.log(10000.0) / (self.d_model // 2))
                pe[i, :, 2 * pos] = torch.sin(x[i, :, 0] * div_term)
                pe[i, :, 2 * pos + 1] = torch.cos(x[i, :, 1] * div_term)

        return pe


class GMIC(nn.Module):
    def __init__(self, parameters):
        super(GMIC, self).__init__()

        # save parameters
        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]

        # construct networks 构建网络
        # global network 全局网络
        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        # shap网络
        # self.shap_network = m.ShapNetwork(self.experiment_parameters, self)
        # self.shap_network.add_layers()

        # changed
        # self.mlp = Mlp(320, 320)

        # aggregation function 聚合函数 TopT百分比聚合函数
        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters, self)

        # detection module 检测模块
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters, self)

        # detection network 检测网络
        self.roitransform = data_augmentation_utils.ROIRotateTransform([0, 90, 180, 270])  # 感兴趣区域增强
        self.local_network = m.LocalNetwork(self.experiment_parameters, self)  # 局部网络
        self.local_network.add_layers()

        # MIL module  多示例模型
        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()

        # fusion branch 融合分支
        # if self.experiment_parameters['learningtype'] == 'SIL':
        # self.fusion_dnn = nn.Linear(parameters["post_processing_dim"] + 512, parameters["num_classes"])

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original 将裁剪位置从cam_size转换为x_original的函数
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version   检索原始图像和小版本的尺寸
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original   将h_small中的二维索引插值到x_original中的索引
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check  健全性检查
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original  将裁剪位置从cam_size插值到x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops  获取原始图像和裁剪位置并返回裁剪的函数
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        # output = torch.ones((batch_size, num_crops, x_original_pytorch.shape[1], crop_h, crop_w))
        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            device = torch.device(self.experiment_parameters["gpu_number"])
            output = output.cuda().to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                gmic_utils.crop_pytorch(x_original_pytorch[i, 0, :, :],
                                        self.experiment_parameters["crop_shape"],
                                        crop_positions[i, j, :],
                                        output[i, j, :, :],
                                        method=crop_method)
        # print("output:", output)
        return output

    def forward(self, x_original, eval_mode):
        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation CNN  类激活映射
        h_g, self.saliency_map, sal_map_before_sigmoid = self.global_network.forward(x_original)

        # image = self.saliency_map[0, 0, :, :].detach().cpu().numpy()
        # img = x_original[0].permute(1, 2, 0).cpu().numpy()
        #
        # resized_image = cv2.resize(image, (896, 2688), interpolation=cv2.INTER_LINEAR)
        # # 绘制图像
        # plt.subplot(1, 2, 1)
        # plt.imshow(img, cmap='gray')
        # plt.title("Original Image")
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img, cmap='gray')
        # plt.imshow(resized_image, cmap='hot', alpha=0.5)
        # plt.title("Saliency CAM")
        # plt.axis('off')
        # plt.show()


        # calculate y_global  评估y_global; note that y_global is not directly used in inference  请注意，在推理中没有直接使用yglobal
        topt_feature, self.y_global = self.aggregation_function.forward(self.saliency_map, sal_map_before_sigmoid)

        # gmic region proposal network  gmic区域提案网络
        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)

        # SHAP
        # h_g, small_x_location, self.saliency_map, sal_map_before_sigmoid = self.global_network.forward(x_original)
        # small_x_locations = small_x_location.cpu().numpy()
        # coords = self.shap_network.forward(x_original)

        # 取ROI点上的feature_map
        # extracted_map = torch.zeros(small_x_locations.shape[0], small_x_locations.shape[1], h_g.shape[1])
        # for i in range(small_x_locations.shape[0]):
        #     for j in range(small_x_locations.shape[1]):
        #         coord = small_x_locations[i, j]
        #         extracted_map[i, j] = h_g[i, :, coord[0], coord[1]]
        # # fm = extracted_map.detach().cuda()
        #
        # global_feature = self.mlp(extracted_map.cuda())

        # 做正则化
        small_x_locations_norm = small_x_locations / self.cam_size

        # convert crop locations that is on self.cam_size to x_original  将self-cam_size上的裁剪位置转换为x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # SHAP
        # coords = np.array(coords)
        # self.patch_locations = np.concatenate((self.patch_locations, coords), axis=1)

        # patch retriever  patch寻回器
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        # detection network  检测网络
        # batch_size, num_crops, Ch, I, J = crops_variable.size()
        batch_size, num_crops, I, J = crops_variable.size()
        # crops_variable = crops_variable.view(batch_size * num_crops, Ch, I, J) #.unsqueeze(1) #60x1x256x256
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        # print("crops_variable:", crops_variable.shape)
        # print(np.moveaxis(x_original[0, :, :, :].cpu().numpy(), 0, -1).shape)
        # plt.imsave(str(0)+'_image.png', x_original[0, 0, :, :].cpu().numpy(), cmap='gray')
        # plt.imshow(x_original[0, 0, :, :].cpu().numpy(), cmap='gray') # 可视化
        # plt.show()

        h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)

        # h_g_l = torch.cat([global_feature, h_crops], dim=1)

        # MIL module
        # y_local is not directly used during inference  在推理过程中不直接使用y_local
        if self.experiment_parameters['learningtype'] == 'SIL':
            z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)
        elif self.experiment_parameters['learningtype'] == 'MIL':
            z, self.patch_attns = self.attention_module.forward(h_crops)
            # z, self.patch_attns = self.attention_module.forward(h_g_l)

        # fusion branch  融合分支
        # use max pooling to collapse the feature CNN   使用最大池来折叠特征图
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)
        concat_vec = torch.cat([global_vec, z], dim=1)

        # if self.experiment_parameters['learningtype'] == 'SIL':
        #     self.y_fusion = self.fusion_dnn(concat_vec)

        if self.experiment_parameters['learningtype'] == 'SIL':
            return self.y_local, self.y_global, self.y_fusion, self.saliency_map, self.patch_locations, self.patches, self.patch_attns, h_crops
        elif self.experiment_parameters['learningtype'] == 'MIL':
            if self.experiment_parameters['model'] == 'gmic_resnet18' or self.experiment_parameters['model'] == 'gmic':
                return z, topt_feature, self.y_global, concat_vec, self.saliency_map, self.patch_locations, self.patches, self.patch_attns, h_crops, global_vec # , topt_feature_before_sig
            else:
                return self.y_fusion

def _gmic(gmic_parameters):
    gmic_model = GMIC(gmic_parameters)

    # if gmic_parameters['pretrained']:
    #     gmic_parameters['arch'] = 'resnet34'
    #     resnet_pretrained_dict = load_state_dict_from_url(model_urls[gmic_parameters['arch']], progress=True)
    #
    #     # load pretrained ImageNet weights for the global network 为全局网络加载预训练的ImageNet权重
    #     gmic_model_dict = gmic_model.state_dict()
    #     global_network_dict = {k: v for (k, v) in gmic_model_dict.items() if 'ds_net' in k}
    #     # 1. filter out unnecessary keys  过滤掉不必要的keys
    #     global_network_pretrained_dict = {'ds_net.' + k: v for k, v in resnet_pretrained_dict.items() if
    #                                       ('ds_net.' + k in global_network_dict) and (k != 'fc.weight') and (
    #                                               k != 'fc.bias')}
    #     # 2. overwrite entries in the existing state dict  覆盖现有状态dict中的条目
    #     gmic_model_dict.update(global_network_pretrained_dict)
    #     print('load res global pretrained')
    #
    #     # load pretrained ImageNet weights for the local network  加载本地网络的预训练ImageNet权重
    #     local_network_dict = {k: v for (k, v) in gmic_model_dict.items() if 'dn_resnet' in k}
    #     # 1. filter out unnecessary keys  过滤掉不必要的keys
    #     local_network_pretrained_dict = {'dn_resnet.' + k: v for k, v in resnet_pretrained_dict.items() if
    #                                      ('dn_resnet.' + k in local_network_dict) and (k != 'fc.weight') and (
    #                                              k != 'fc.bias')}
    #     # 2. overwrite entries in the existing state dict  覆盖现有状态dict中的条目
    #     gmic_model_dict.update(local_network_pretrained_dict)
    #
    #     gmic_model.load_state_dict(gmic_model_dict)
    #     print('load res local pretrained')

    return gmic_model
