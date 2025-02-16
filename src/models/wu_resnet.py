# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3

from src.models.Coc import coc_tiny
from src.models.sparsevit_gmic import SparseViT
from src.models.swin_transformer import SwinTransformer


def load_pretrained(model, pretrained):
    print(f"==============> Loading weight {pretrained} for fine-tuning......")
    checkpoint = torch.load(pretrained, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if L1 != L2:
            # bicubic interpolate relative_position_bias_table if not match
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                mode='bicubic')
            state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
            absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
            absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
            absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
            state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)

    print(f"=> loaded successfully '{pretrained}'")

    del checkpoint
    torch.cuda.empty_cache()


class SplitBreastModel(nn.Module):
    def __init__(self, config_params):
        super(SplitBreastModel, self).__init__()

        self.channel = config_params['channel']

        self.four_view_resnet = FourViewResNet(self.channel, config_params)

        self.fc1_cc = nn.Linear(768 * 2, 768)
        self.fc1_mlo = nn.Linear(768 * 2, 768)
        self.output_layer_cc = OutputLayer(768, 1)
        self.output_layer_mlo = OutputLayer(768, 1)

        self.all_views_avg_pool = AllViewsAvgPool()
        self.view2views = ['LCC', 'LMLO', 'RCC', 'RMLO']

    def forward(self, x, views_names, eval_mode):
        result = self.four_view_resnet(x, views_names)
        h = self.all_views_avg_pool(result)

        # changed
        if len(views_names) == 4:
            # Pool, flatten, and fully connected layers
            h_cc = torch.cat([h['LCC'], h['RCC']], dim=1)
            h_mlo = torch.cat([h['LMLO'], h['RMLO']], dim=1)

            h_cc = F.relu(self.fc1_cc(h_cc))
            h_mlo = F.relu(self.fc1_mlo(h_mlo))

            h_cc = self.output_layer_cc(h_cc)
            h_mlo = self.output_layer_mlo(h_mlo)
            out = torch.cat((h_cc, h_mlo), dim=1)
            out = torch.mean(out, dim=1)
        elif len(views_names) == 3:
            if self.view2views[0] in views_names and self.view2views[2] in views_names:
                h_cc = torch.cat([h['LCC'], h['RCC']], dim=1)
                h_cc = F.relu(self.fc1_cc(h_cc))
                h_cc = self.output_layer_cc(h_cc)
                if self.view2views[1] in views_names:
                    h_mlo = self.output_layer_mlo(h['LMLO'])
                if self.view2views[3] in views_names:
                    h_mlo = self.output_layer_mlo(h['RMLO'])
                out = torch.cat((h_cc, h_mlo), dim=1)
                out = torch.mean(out, dim=1)
            elif self.view2views[1] in views_names and self.view2views[3] in views_names:
                h_mlo = torch.cat([h['LMLO'], h['RMLO']], dim=1)
                h_mlo = F.relu(self.fc1_mlo(h_mlo))
                h_mlo = self.output_layer_mlo(h_mlo)
                if self.view2views[0] in views_names:
                    h_cc = self.output_layer_cc(h['LCC'])
                if self.view2views[2] in views_names:
                    h_cc = self.output_layer_cc(h['RCC'])
                out = torch.cat((h_cc, h_mlo), dim=1)
                out = torch.mean(out, dim=1)
        elif len(views_names) == 2:
            if self.view2views[0] in views_names and self.view2views[2] in views_names:
                h_cc = torch.cat([h['LCC'], h['RCC']], dim=1)
                h_cc = F.relu(self.fc1_cc(h_cc))
                h_cc = self.output_layer_cc(h_cc)
                out = torch.mean(h_cc, dim=1)
            elif self.view2views[0] in views_names and self.view2views[1] in views_names:
                h_cc = self.output_layer_cc(h['LCC'])
                h_mlo = self.output_layer_mlo(h['LMLO'])
                out = torch.cat((h_cc, h_mlo), dim=1)
                out = torch.mean(out, dim=1)
            elif self.view2views[0] in views_names and self.view2views[3] in views_names:
                h_cc = self.output_layer_cc(h['LCC'])
                h_mlo = self.output_layer_mlo(h['RMLO'])
                out = torch.cat((h_cc, h_mlo), dim=1)
                out = torch.mean(out, dim=1)
            elif self.view2views[1] in views_names and self.view2views[3] in views_names:
                h_mlo = torch.cat([h['LMLO'], h['RMLO']], dim=1)
                h_mlo = F.relu(self.fc1_mlo(h_mlo))
                h_mlo = self.output_layer_mlo(h_mlo)
                out = torch.mean(h_mlo, dim=1)
            elif self.view2views[1] in views_names and self.view2views[2] in views_names:
                h_cc = self.output_layer_cc(h['RCC'])
                h_mlo = self.output_layer_mlo(h['LMLO'])
                out = torch.cat((h_cc, h_mlo), dim=1)
                out = torch.mean(out, dim=1)
            elif self.view2views[2] in views_names and self.view2views[3] in views_names:
                h_cc = self.output_layer_cc(h['RCC'])
                h_mlo = self.output_layer_mlo(h['RMLO'])
                out = torch.cat((h_cc, h_mlo), dim=1)
                out = torch.mean(out, dim=1)
        else:
            view = next(iter(h))
            out = torch.mean(h[view], dim=1)
        return out


class FourViewResNet(nn.Module):
    def __init__(self, input_channels, config_params):
        super(FourViewResNet, self).__init__()
        # self.cc = resnet22(input_channels)
        # self.mlo = resnet22(input_channels)
        # self.cc = SwinTransformer(img_size=[2688, 896])
        # self.mlo = SwinTransformer(img_size=[2688, 896])

        # self.cc = coc_tiny(pretrained='/home/kemove/下载/model_best_tiny.pth.tar')
        # self.mlo = coc_tiny(pretrained='/home/kemove/下载/model_best_tiny.pth.tar')

        # swin-transformer
        # load_pretrained(self.cc, pretrained=r'/home/kemove/下载/swin_tiny_patch4_window7_224.pth')
        # load_pretrained(self.mlo, pretrained=r'/home/kemove/下载/swin_tiny_patch4_window7_224.pth')
        # for param in self.cc.parameters():
        #     param.requires_grad = False
        # # 解冻最后一层的参数
        # for param in self.cc.layers[-1].parameters():
        #     param.requires_grad = True
        #
        # for param in self.mlo.parameters():
        #     param.requires_grad = False
        # # 解冻最后一层的参数
        # for param in self.mlo.layers[-1].parameters():
        #     param.requires_grad = True

        self.cc = SparseViT(init_cfg='/home/kemove/下载/mask_rcnn_sparsevit_cfg1_42ms.pth', pruning_ratios=[[0.3, 0.3], [0., 0.], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [0., 0.]])
        self.mlo = SparseViT(init_cfg='/home/kemove/下载/mask_rcnn_sparsevit_cfg1_42ms.pth', pruning_ratios=[[0.3, 0.3], [0., 0.], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [0., 0.]])

        for param in self.cc.parameters():
            param.requires_grad = False
        #
        # 解冻最后一层的参数
        for param in self.cc.stages[-1].parameters():
            param.requires_grad = True
        for param in self.cc.stages[-2].parameters():
            param.requires_grad = True

        for param in self.mlo.parameters():
            param.requires_grad = False
        #
        # 解冻最后一层的参数
        for param in self.mlo.stages[-1].parameters():
            param.requires_grad = True
        for param in self.mlo.stages[-2].parameters():
            param.requires_grad = True

        self.model_dict = {}
        self.model_dict['LCC'] = self.l_cc = self.cc
        self.model_dict['LMLO'] = self.l_mlo = self.mlo
        self.model_dict['RCC'] = self.r_cc = self.cc
        self.model_dict['RMLO'] = self.r_mlo = self.mlo

        for k, v in self.cc.named_parameters():
            print('{}: {}'.format(k, v.requires_grad))

    def forward(self, x, views_names):
        h_dict = {
            view: self.single_forward(x[:, views_names.index(view), :, :].unsqueeze(1), view)
            for view in views_names
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.model_dict[view](single_x.squeeze(1))


class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        # if not isinstance(output_shape, (list, tuple)):
        #    output_shape = [output_shape]
        self.output_shape = output_shape
        # self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.output_shape)  # self.flattened_output_shape)

    def forward(self, x):
        h = self.fc_layer(x)
        # if len(self.output_shape) > 1:
        h = h.view(h.shape[0], self.output_shape)
        # h = F.log_softmax(h, dim=-1)
        return h


class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        return {
            'LCC': self.single_add_gaussian_noise(x['LCC']),
            'LMLO': self.single_add_gaussian_noise(x['LMLO']),
            'RCC': self.single_add_gaussian_noise(x['RCC']),
            'RMLO': self.single_add_gaussian_noise(x['RMLO']),
        }

    def single_add_gaussian_noise(self, single_view):
        if not self.gaussian_noise_std or not self.training:
            return single_view
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            view_name: self.single_avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }

    @staticmethod
    def single_avg_pool(single_view):
        n, c, _, _ = single_view.size()
        return single_view.view(n, c, -1).mean(-1)


class BasicBlockV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
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


class ViewResNetV2(nn.Module):
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
        super(ViewResNetV2, self).__init__()
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


def resnet22(input_channels):
    return ViewResNetV2(
        input_channels=input_channels,
        num_filters=16,
        first_layer_kernel_size=7,
        first_layer_conv_stride=2,
        blocks_per_layer_list=[2, 2, 2, 2, 2],
        block_strides_list=[1, 2, 2, 2, 2],
        block_fn=BasicBlockV2,
        first_layer_padding=0,
        first_pool_size=3,
        first_pool_stride=2,
        first_pool_padding=0,
        growth_factor=2
    )
