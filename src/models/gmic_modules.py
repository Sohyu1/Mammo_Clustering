
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.swin_transformer import SwinTransformer
from src.models.wu_resnet import load_pretrained
from src.utilities import gmic_utils
from torchvision.models.resnet import conv3x3
from src.models.Coc.context_cluster import coc_tiny, coc_tiny_plain
from src.models.sparsevit_gmic import SparseViT
from src.models.WTConv.wtconvnext import WTConvNeXt, checkpoint_filter_fn


class BasicBlockV2(nn.Module):
    """
    Basic Residual Block of ResNet V2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class BasicBlockV1(nn.Module):
    """
    Basic Residual Block of ResNet V1 (ResNet18)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """

    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * block_fn.expansion
        )
        self.relu = nn.ReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)


class ResNetV1(nn.Module):
    """
    Class that represents a ResNet with classifier sequence removed
    """

    def __init__(self, initial_filters, block, layers, input_channels=1):

        self.inplanes = initial_filters
        self.num_layers = len(layers)
        super(ResNetV1, self).__init__()

        # initial sequence
        # the first sequence only has 1 input channel which is different from original ResNet
        self.conv1 = nn.Conv2d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual sequence
        for i in range(self.num_layers):
            num_filters = initial_filters * pow(2, i)
            num_stride = (1 if i == 0 else 2)
            setattr(self, 'layer{0}'.format(i + 1), self._make_layer(block, num_filters, layers[i], stride=num_stride))
        self.num_filter_last_seq = initial_filters * pow(2, self.num_layers - 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # first sequence
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual sequences
        for i in range(self.num_layers):
            x = getattr(self, 'layer{0}'.format(i + 1))(x)
        return x


class DownsampleNetworkResNet18V1(ResNetV1):
    """
    Downsampling using ResNet V1
    First conv is 7*7, stride 2, padding 3, cut 1/2 resolution  第一个conv是7*7，步幅2，填充3，剪切1/2分辨率
    """

    def __init__(self):
        super(DownsampleNetworkResNet18V1, self).__init__(
            initial_filters=64,
            block=BasicBlockV1,
            layers=[2, 2, 2, 2],
            input_channels=3)

    def forward(self, x):
        last_feature_map = super(DownsampleNetworkResNet18V1, self).forward(x)
        return last_feature_map


class DownsampleNetworkResNet34V1(ResNetV1):
    """
    Downsampling using ResNet V1
    First conv is 7*7, stride 2, padding 3, cut 1/2 resolution  第一个conv是7*7，步幅2，填充3，剪切1/2分辨率
    """

    def __init__(self):
        super(DownsampleNetworkResNet34V1, self).__init__(
            initial_filters=64,
            block=BasicBlockV1,
            layers=[3, 4, 6, 3],
            input_channels=3)

    def forward(self, x):
        last_feature_map = super(DownsampleNetworkResNet34V1, self).forward(x)
        return last_feature_map


class AbstractMILUnit:
    """
    An abstract class that represents an MIL unit module  表示MIL单元模块的抽象类
    """

    def __init__(self, parameters, parent_module):
        self.parameters = parameters
        self.parent_module = parent_module


class PostProcessingStandard(nn.Module):
    """
    Unit in Global Network that takes in x_out and produce Atten maps   Global Network中接收x_out并生成显著性图SM（特征图）的单元
    """

    def __init__(self, parameters):
        super(PostProcessingStandard, self).__init__()
        # CNN all filters to output classes  将所有filters映射到输出类
        # changed
        parameters["post_processing_dim"] = 768  # 320, 768
        self.gn_conv_last = nn.Conv2d(parameters["post_processing_dim"],
                                      parameters["num_classes"],
                                      (1, 1), bias=False)

    def forward(self, x_out):
        out = self.gn_conv_last(x_out)
        return out, torch.sigmoid(out)


class GlobalNetwork(AbstractMILUnit):
    """
    Implementation of Global Network using ResNet-22
    """

    def __init__(self, parameters, parent_module):
        super(GlobalNetwork, self).__init__(parameters, parent_module)
        # changed
        # VMamba
        # self.downsampling_branch = Backbone_VSSM(out_indices=[3])

        # Resnet
        # self.downsampling_branch = DownsampleNetworkResNet18V1()  # 下采样
        # self.downsampling_branch = DownsampleNetworkResNet34V1()  # 下采样

        # Coc
        # self.downsampling_branch = coc_tiny(pretrained=r'D:\Code\Python_Code\Mammo\datasets\model_best.pth.tar')

        # SparseVit
        # self.downsampling_branch = SparseViT(init_cfg='/home/kemove/下载/mask_rcnn_sparsevit_cfg1_42ms.pth', pruning_ratios=[[0.3, 0.3], [0., 0.], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [0., 0.]])

        # SwinTransformer
        self.downsampling_branch = SwinTransformer(img_size=[2688, 896])
        load_pretrained(self.downsampling_branch, pretrained=r'D:\Code\Python_Code\Mammo\datasets\swin_tiny_patch4_window7_224.pth')


        # load pretrain_model  coc network
        for param in self.downsampling_branch.parameters():
            param.requires_grad = False
        # 解冻最后一层的参数
        for param in self.downsampling_branch.layers[-1].parameters():
            param.requires_grad = True
        for param in self.downsampling_branch.layers[-2].parameters():
            param.requires_grad = True

        # for k, v in self.downsampling_branch.network.named_parameters():
        #     print('{}: {}'.format(k, v.requires_grad))

        # post-processing  后处理
        self.postprocess_module = PostProcessingStandard(parameters)

    def add_layers(self):
        self.parent_module.ds_net = self.downsampling_branch
        self.parent_module.left_postprocess_net = self.postprocess_module

    # # Shap
    # def get_block_coordinates(self, block_number, block_size=(7, 7), tensor_size=(84, 28)):
    #     # 计算每一维度的块数
    #     num_blocks_row = tensor_size[0] // block_size[0]
    #     num_blocks_col = tensor_size[1] // block_size[1]
    #
    #     # 计算块所在的行号和列号
    #     row_index = block_number // num_blocks_col
    #     col_index = block_number % num_blocks_col
    #
    #     # 计算块在原始 tensor 中的起始坐标
    #     start_row = row_index * block_size[0]
    #     start_col = col_index * block_size[1]
    #
    #     location = torch.stack((start_row, start_col), dim=-1)
    #     return location

    def forward(self, x):
        # retrieve results from downsampling network at all 4 levels  从所有4个级别的下采样网络中检索结果 做出特征图
        last_feature_map = self.downsampling_branch.forward(x)

        # feed into postprocessing network  馈入后处理网络  映射到num-class
        cam_before_sigmoid, cam = self.postprocess_module.forward(last_feature_map)
        return last_feature_map, cam, cam_before_sigmoid

        # # Shap
        # last_feature_map, indexs = self.downsampling_branch.forward(x)
        # indexs_a = indexs[:, :self.K]
        # locations = self.get_block_coordinates(indexs_a)
        # return last_feature_map, locations, cam, cam_before_sigmoid


class TopTPercentAggregationFunction(AbstractMILUnit):
    """
    An aggregator that uses the SM to compute the y_global.  使用SM（Atten maps）计算y_global的聚合器。
    Use the sum of topK value  使用topK值的总和
    """

    def __init__(self, parameters, parent_module):
        super(TopTPercentAggregationFunction, self).__init__(parameters, parent_module)
        self.percent_t = parameters["percent_t"]
        self.parent_module = parent_module

    def forward(self, cam, cam_before_sig):
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W * H * self.percent_t))  # round函数用来返回一个浮点数的四舍五入值
        selected_area, topt_index = cam_flatten.topk(top_t, dim=2)  # topt_index: N, 1, 110
        # selectedarea_before_sig = torch.gather(cam_before_sig.view(batch_size, num_class, -1), 2, topt_index) #N, 1, 5520
        return selected_area, selected_area.mean(dim=2)


class RetrieveROIModule(AbstractMILUnit):
    """
    A Regional Proposal Network instance that computes the locations of the crops   v
    Greedy select crops with largest sums  贪婪地选择sums最大的裁剪图
    """

    def __init__(self, parameters, parent_module):
        super(RetrieveROIModule, self).__init__(parameters, parent_module)
        self.crop_method = "upper_left"
        self.num_crops_per_class = parameters["K"]
        self.crop_shape = parameters["crop_shape"]
        self.gpu_number = None if parameters["device_type"] != "gpu" else parameters["gpu_number"]

    def forward(self, x_original, cam_size, h_small):
        """
        Function that use the low-res image to determine the position of the high-res crops  使用低分辨率图像确定高分辨率裁剪图位置的函数
        :param x_original: N, C, H, W pytorch tensor
        :param cam_size: (h, w) -> Atten CNN height, width
        :param h_small: N, C, h_h, w_h pytorch tensor
        :return: N, num_classes*k, 2 numpy matrix; returned coordinates are corresponding to x_small
        """
        # retrieve parameters  检索参数
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()
        # print(h,h_h)
        # print(w,w_h)

        # make sure that the size of h_small == size of cam_size  确保h_small的大小==cam_size的大小
        assert h_h == h, "h_h!=h"
        assert w_h == w, "w_h!=w"

        # adjust crop_shape since crop shape is based on the original image  调整crop_shape，因为裁剪形状基于原始图像
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        # greedily find the box with max sum of weights  贪婪找权重总和最大的box
        current_images = h_small
        all_max_position = []
        # combine channels  组合通道
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)

        for _ in range(self.num_crops_per_class):
            max_pos = gmic_utils.get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = gmic_utils.generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, self.gpu_number)
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()


class LocalNetwork(AbstractMILUnit):
    """
    The local network that takes a crop and computes its hidden representation  获取裁剪图并计算其隐藏表示的本地网络
    Use ResNet
    """

    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        # changed
        # VMamba
        # self.parent_module.dn_resnet = Backbone_VSSM(out_indices=[3])

        # Resnet
        # self.parent_module.dn_resnet = ResNetV1(64, BasicBlockV1, [2, 2, 2, 2], 3)
        # self.parent_module.dn_resnet = ResNetV1(64, BasicBlockV1, [3, 4, 6, 3], 3)

        # Coc
        self.parent_module.dn_resnet = coc_tiny(pretrained=r'D:\Code\Python_Code\Mammo\datasets\model_best.pth.tar')

        # SparseVit
        # self.parent_module.dn_resnet = SparseViT(init_cfg='/home/kemove/下载/mask_rcnn_sparsevit_cfg1_42ms.pth', pruning_ratios=[[0.3, 0.3], [0., 0.], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [0., 0.]])

        # wtconv
        # self.parent_module.dn_resnet = WTConvNeXt()
        # model_weights = torch.load(r'D:\Code\Python_Code\Mammo\datasets\WTConvNeXt_tiny_5_300e_ema.pth')
        # self.parent_module.dn_resnet.load_state_dict(model_weights)

        # load pretrain_model
        for param in self.parent_module.dn_resnet.parameters():
            param.requires_grad = False
        #
        # # 解冻最后一层的参数  coc network
        for param in self.parent_module.dn_resnet.network[-1].parameters():
            param.requires_grad = True
        for param in self.parent_module.dn_resnet.network[-2].parameters():
            param.requires_grad = True


    def forward(self, x_crop):
        """
        Function that takes in a single crop and return the hidden representation
        :param x_crop: (N,C,h,w)
        :return:
        """
        # forward propagte using ResNet
        res = self.parent_module.dn_resnet(x_crop.expand(-1, 3, -1, -1))
        # res, _ = self.parent_module.dn_resnet(x_crop.expand(-1, 3, -1, -1))

        # torch.save(self.parent_module.dn_resnet.state_dict(),
        #            '/home/kemove/PycharmProjects/multiinstance-learning-mammography/src/out_res/run/model_local.tar')
        # res = self.parent_module.dn_resnet(x_crop)

        # global average pooling
        res = res.mean(dim=2).mean(dim=2)
        return res


class AttentionModule(AbstractMILUnit):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average   注意力模块采用多个隐藏表示并计算注意力加权平均值
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf   使用门控注意力机制
    """

    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module  给nn.module的父模块添加层
        :return:
        """
        # The gated attention mechanism  门控注意力机制
        # changed  # 320, 768
        self.parent_module.mil_attn_V = nn.Linear(320, 128, bias=False)
        self.parent_module.mil_attn_U = nn.Linear(320, 128, bias=False)
        self.parent_module.mil_attn_w = nn.Linear(128, 1, bias=False)

    def forward(self, h_crops):
        """
        Function that takes in the hidden representations of crops and use attention to generate a single hidden vector  接收裁剪图的隐藏表示并使用注意力生成单个隐藏向量的函数
        :param h_small:
        :param h_crops:
        :return:
        """
        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        # calculate the attn score  计算注意力分数
        attn_projection = torch.sigmoid(self.parent_module.mil_attn_U(h_crops_reshape)) * \
                          torch.tanh(self.parent_module.mil_attn_V(h_crops_reshape))
        attn_score = self.parent_module.mil_attn_w(attn_projection)
        # use softmax to CNN score to attention  使用softmax将分数映射到注意力
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)

        # final hidden vector  最终隐藏向量
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)

        # CNN to the final layer  映射到最终隐藏向量
        # y_crops = torch.sigmoid(self.parent_module.classifier_linear(z_weighted_avg))
        if self.parameters['learningtype'] == 'SIL':
            y_crops = self.parent_module.classifier_linear(z_weighted_avg)
            return z_weighted_avg, attn, y_crops
        elif self.parameters['learningtype'] == 'MIL':
            return z_weighted_avg, attn



class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
