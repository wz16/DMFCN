import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import math
import numpy as np
from typing import cast, Union, List


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                128, configs.c_out, bias=True)
            output_dim = configs.d_model
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
            output_dim = configs.c_out
        if self.task_name == 'classification':
            # self.act = F.gelu
            # self.dropout = nn.Dropout(configs.dropout)
            # self.projection = nn.Linear(
            #     configs.d_model * configs.seq_len, configs.num_class)
            output_dim = configs.num_class
        if self.task_name == 'regression':
            # self.act = F.gelu
            # self.dropout = nn.Dropout(configs.dropout)
            # self.projection = nn.Linear(
            #     configs.d_model * configs.seq_len, 1)
            output_dim = 1
        base_input_dim = configs.enc_in
        self.n_infer_steps = configs.seq_len

        input_shape = base_input_dim
        hidden_dim = 32
        self.Inception = Inception(base_input_dim*self.n_infer_steps, base_input_dim, hidden_dim, output_dim)
        self.hidden = nn.Linear(hidden_dim*4, output_dim)
        self.averagepool = nn.AdaptiveAvgPool1d(1)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # # embedding
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
        #     0, 2, 1)  # align temporal dimension
        # # TimesNet
        # for i in range(self.layer):
        #     enc_out = self.layer_norm(self.model[i](enc_out))
        # # porject back
        # dec_out = self.projection(enc_out)

        h = x_enc
        h = self.predict_linear(h.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.Inception(h.transpose(1,2)).transpose(1,2)
        # output = self.predict_linear(output)
        dec_out = self.projection(output)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        h = x_enc
        X = self.Inception(h.transpose(1,2)).transpose(1,2)
        dec_out = self.hidden(X)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        h = x_enc
        X = self.Inception(h.transpose(1,2)).transpose(1,2)
        dec_out = self.hidden(X)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        h = x_enc
        X = self.Inception(h)
        X = self.averagepool(X)
        X = X.squeeze_(-1)
        X = self.hidden(X)

        return X
    def regression(self, x_enc, x_mark_enc):
        h = x_enc
        X = self.Inception(h)

        X = self.averagepool(X)
        X = X.squeeze_(-1)
        X = self.hidden(X)

        return X
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'regression':
            dec_out = self.regression(x_enc, x_mark_enc)
            return dec_out  # [B, 1]
        return None



class Inception(nn.Module):

    
    """
    Modified from https://github.com/okrasolar/pytorch-timeseries and https://github.com/timeseriesAI/tsai

    A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    input_dim:
        The number of input channels (i.e. input.shape[-1])
    hidden_dim:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    output_dim:
        The number of output classes
    """

    def __init__(self, input_dim: int, base_input_dim, hidden_dim, output_dim,
                 num_blocks = 6, bottleneck_channels: Union[List[int], int] = 32 , kernel_sizes: Union[List[int], int] = 41,
                 use_residuals: Union[List[bool], bool, str] = 'default', **kwargs) :
        super().__init__()
        self.n_infer_steps = input_dim//base_input_dim
        input_dim = base_input_dim       
        # for easier saving and loading
        self.input_args = {
            'num_blocks': num_blocks,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'output_dim': output_dim
        }

        # channels = [input_dim] + cast(List[int], self._expand_to_blocks(hidden_dim,
        #                                                                   num_blocks))
        # bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
        #                                                              num_blocks))
        # kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
        )

        self.blocks = nn.Sequential(*[
            InceptionBlock(input_dim=input_dim if i == 0 else 4*hidden_dim, hidden_dim=hidden_dim,
                           residual=use_residuals[i], bottleneck_channels=bottleneck_channels,
                           kernel_size=kernel_sizes) for i in range(num_blocks)
        ])

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.linear = nn.Linear(in_features=4*hidden_dim, out_features=output_dim)
        # summary(self, input_size=(18, self.n_infer_steps, base_input_dim))


        

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if len(x.shape)== 2:
            x = torch.reshape(x, (x.shape[0], self.base_input_dim, -1))
        x = x.transpose(2,1)
        if x.shape[2]>self.n_infer_steps:
            x = x[:,:,-self.n_infer_steps:]
        x = self.blocks(x) # the mean is the global average pooling
        return x

class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(input_dim, bottleneck_channels,
                                                kernel_size=1, bias=False)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else input_dim
        channels = [start_channels] + [hidden_dim] * 3
        self.conv_layers = nn.ModuleList([Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=stride, bias=False)
            for i in range(len(kernel_size_s))])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool1d(kernel_size= 3, padding = 1, stride=stride)
        self.conv_6 = Conv1dSamePadding(input_dim, bottleneck_channels,
                                                kernel_size=1, bias=False)

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=input_dim, out_channels=hidden_dim*4,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(hidden_dim*4),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        xs = [conv_layer(x) for conv_layer in self.conv_layers]
        max_pool = self.max_pool(org_x)
        xs.append(self.conv_6(max_pool))
        x = torch.cat(xs, dim = 1)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x



class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)

def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)