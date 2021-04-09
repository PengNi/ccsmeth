#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from utils.constants_torch import use_cuda

# import math


# BiLSTM ===============================================================
class ModelRNN(nn.Module):
    def __init__(self, seq_len=21, num_layers=3, num_classes=2,
                 dropout_rate=0.5, hidden_size=256,
                 vocab_size=16, embedding_size=4,
                 is_stds=True,
                 model_type="bilstm"):
        super(ModelRNN, self).__init__()

        self.model_type = model_type

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base

        self.is_stds = is_stds
        self.feas_ccs = 4 if self.is_stds else 2
        if self.model_type == "bilstm":
            self.rnn = nn.LSTM(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=dropout_rate, batch_first=True, bidirectional=True)
        elif self.model_type == "bigru":
            self.rnn = nn.GRU(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=dropout_rate, batch_first=True, bidirectional=True)
        else:
            raise ValueError("model_type not set right!")

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)  # 2 for bidirection
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    def forward(self, kmer, ipd_means, ipd_stds, pw_means, pw_stds):
        # kmer, ipd means, ipd_stds, pw_means, pw_stds as features
        kmer_embed = self.embed(kmer.long())
        ipd_means = torch.reshape(ipd_means, (-1, self.seq_len, 1)).float()
        pw_means = torch.reshape(pw_means, (-1, self.seq_len, 1)).float()
        if self.is_stds:
            ipd_stds = torch.reshape(ipd_stds, (-1, self.seq_len, 1)).float()
            pw_stds = torch.reshape(pw_stds, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((kmer_embed, ipd_means, ipd_stds, pw_means, pw_stds), 2)  # (N, L, C)
        else:
            out1 = torch.cat((kmer_embed, ipd_means, pw_means), 2)  # (N, L, C)
        out1, _ = self.rnn(out1, self.init_hidden(out1.size(0),
                                                  self.num_layers,
                                                  self.hidden_size))  # (N, L, nhid*2)
        # decode
        out1_fwd_last = out1[:, -1, :self.hidden_size]
        out1_bwd_last = out1[:, 0, self.hidden_size:]
        out = torch.cat((out1_fwd_last, out1_bwd_last), 1)

        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out, self.softmax(out)


# BiLSTM ===============================================================
class ModelAttRNN(nn.Module):
    def __init__(self, seq_len=21, num_layers=3, num_classes=2,
                 dropout_rate=0.5, hidden_size=256,
                 vocab_size=16, embedding_size=4,
                 is_stds=True,
                 model_type="attbilstm"):
        super(ModelAttRNN, self).__init__()
        self.model_type = model_type

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base

        self.is_stds = is_stds
        self.feas_ccs = 4 if self.is_stds else 2
        if self.model_type == "attbilstm":
            self.rnn = nn.LSTM(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=dropout_rate, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru":
            self.rnn = nn.GRU(embedding_size + self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=dropout_rate, batch_first=True, bidirectional=True)
        else:
            raise ValueError("model_type not set right!")

        # self.dropout1 = nn.Dropout(p=dropout_rate)
        # self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)  # 2 for bidirection
        # self.relu = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=dropout_rate)
        # self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.num_classes)  # 2 for bidirection

        # for attention_net2
        self._att2_proj = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self._att2_tanh = nn.Tanh()
        self._att2_context_vector = nn.Parameter(torch.Tensor(self.hidden_size * 2, 1))
        self._att2_softmax = nn.Softmax(dim=1)

        # for attention_net3
        self._att3_proj = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self._att3_tanh = nn.Tanh()
        self._att3_softmax = nn.Softmax(dim=1)

        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    # https://github.com/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi-LSTM(Attention).py
    # https://github.com/zhijing-jin/pytorch_RelationExtraction_AttentionBiLSTM/blob/master/model.py
    def attention_net(self, rnn_output, final_state):
        # lstm_output : [batch_size, seq_len, n_hidden * num_directions(=2)]
        # final_state : [2, batch_size, n_hid]
        hidden = final_state.view(-1, self.hidden_size * 2, 1)
        # hidden = final_state.permute(1, 0, 2).reshape(-1, self.hidden_size * 2, 1)  # (N, 2 * n_hid, 1)

        attn_weights = torch.bmm(rnn_output, hidden).squeeze(2)  # attn_weights : [batch_size, seq_len]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), seq_len] * [batch_size, seq_len] =
        # [batch_size, n_hidden * num_directions(=2), 1]
        attout = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # attout: [batch_size, n_hidden * num_directions(=2)]
        return attout

    # https://www.dazhuanlan.com/2019/12/16/5df6a2a0b9dde/
    # https://github.com/Cheneng/HiararchicalAttentionGRU/blob/master/model/HiararchicalATT.py
    # https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/word_att_model.py
    def attention_net2(self, rnn_output):
        # lstm_output : [N, L, n_hid * 2]
        Hw = self._att2_tanh(self._att2_proj(rnn_output))  # (N, L, n_hid * 2)
        w_score = self._att2_softmax(Hw.matmul(self._att2_context_vector))  # (N, L, 1)
        # return torch.mul(rnn_output, w_score)  # (N, L, n_hid * 2)
        return torch.bmm(rnn_output.transpose(1, 2), w_score).squeeze(2)  # (N, n_hid * 2)

    def attention_net3(self, rnn_output, final_state):
        # final_state : [2, batch_size, n_hid]
        hidden = final_state.view(-1, self.hidden_size * 2, 1)
        # hidden = final_state.permute(1, 0, 2).reshape(-1, self.hidden_size * 2, 1)  # (N, 2 * n_hid, 1)

        # lstm_output : [N, L, n_hid * 2]
        Hw = self._att3_tanh(self._att3_proj(rnn_output))  # (N, L, n_hid * 2)
        w_score = self._att3_softmax(Hw.matmul(hidden))  # (N, L, 1)
        return torch.bmm(rnn_output.transpose(1, 2), w_score).squeeze(2)  # (N, n_hid * 2)

    def forward(self, kmer, ipd_means, ipd_stds, pw_means, pw_stds):
        # kmer, ipd means, ipd_stds, pw_means, pw_stds as features
        kmer_embed = self.embed(kmer.long())
        ipd_means = torch.reshape(ipd_means, (-1, self.seq_len, 1)).float()
        pw_means = torch.reshape(pw_means, (-1, self.seq_len, 1)).float()
        if self.is_stds:
            ipd_stds = torch.reshape(ipd_stds, (-1, self.seq_len, 1)).float()
            pw_stds = torch.reshape(pw_stds, (-1, self.seq_len, 1)).float()
            out1 = torch.cat((kmer_embed, ipd_means, ipd_stds, pw_means, pw_stds), 2)  # (N, L, C)
        else:
            out1 = torch.cat((kmer_embed, ipd_means, pw_means), 2)  # (N, L, C)

        out, (h_n, c_n) = self.rnn(out1, self.init_hidden(out1.size(0),
                                                          self.num_layers,
                                                          self.hidden_size))  # (N, L, nhid*2)

        # attention_net ======
        # h_n: (num_layer * 2, N, nhid), h_0, c_0 -> h_n, c_n not affected by batch_first
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        out = self.attention_net(out, h_n)

        # # attention_net2 ======
        # out = self.attention_net2(out)

        # # attention_net3 ======
        # h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        # out = self.attention_net3(out, h_n)

        # out = self.dropout1(out)
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.dropout2(out)
        # out = self.fc2(out)

        out = self.dropout1(out)
        out = self.fc1(out)

        return out, self.softmax(out)


# ResNet ===============================================================
# https://github.com/pytorch/vision/blob/7c077f6a986f05383bcb86b535aedb5a63dd5c4b/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, init_channels=2, num_classes=2, dropout_rate=0.5,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(init_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(init_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # X = [N, C, Hi, Wi]
        x = x.float()
        # print("input shape: {}".format(x.size()))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print("============================before layer1: {}".format(x.size()))
        x = self.layer1(x)
        # print("============================layer1: {}".format(x.size()))
        x = self.layer2(x)
        # print("============================layer2: {}".format(x.size()))
        x = self.layer3(x)
        # print("============================layer3: {}".format(x.size()))
        x = self.layer4(x)
        # print("============================layer4: {}".format(x.size()))

        x = self.avgpool(x)
        # print("============================avgpool: {}".format(x.size()))
        x = torch.flatten(x, 1)
        # print("============================flatten: {}".format(x.size()))
        x = self.dropout(x)
        # x = self.fc(x)
        # print("============================output: {}\n".format(x.size()))

        # return x, self.softmax(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


class ModelResNet18(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5, is_stds=True):
        super(ModelResNet18, self).__init__()

        self.model_type = "ResNet18"

        self.num_classses = num_classes

        self.is_stds = is_stds
        channels_ccs = 4 if self.is_stds else 2
        self.resnet18_ccs = resnet18(init_channels=channels_ccs, num_classes=self.num_classses,
                                     dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(512 * BasicBlock.expansion, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, self.num_classses)

        self.softmax = nn.Softmax(1)

    def forward(self, mat_ccs_mean, mat_ccs_std):
        if self.is_stds:
            out1 = torch.cat((mat_ccs_mean, mat_ccs_std), 1).float()  # (N, C=4, H, W)
        else:
            out1 = mat_ccs_mean.float()  # (N, C=2, H, W)
        out = self.resnet18_ccs(out1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)

        return out, self.softmax(out)
