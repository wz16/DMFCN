import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

class Linear_variableaxis(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear_variableaxis, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        return
    def forward(self, x):
        # x = torch.transpose(x,1,2)
        x = self.linear(x)
        # x = torch.transpose(x,1,2)
        return x

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class DMFCN_Block(nn.Module):
    def __init__(self, configs, if_first_layer=False):
        super(DMFCN_Block, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        filters1 = [3,8,8,8]
        dilation1 = [1,1,5,10]
        filters2 = [3,8,8,8]
        dilation2 = [1,1,5,10]
        if if_first_layer:
            base_dim = configs.enc_in
        else:
            base_dim = configs.d_model

        second_layer_channel, third_layer_channel = 128, 256
        layer_parameter_lists = [[(base_dim, second_layer_channel//4, filters1[i],dilation1[i]) for i in range(len(filters1))],
        [(second_layer_channel, third_layer_channel//4, filters2[i],dilation2[i]) for i in range(len(filters2))],
        [(third_layer_channel, configs.d_model//4, 1, 1),(third_layer_channel, configs.d_model//4, 1, 1),(third_layer_channel, configs.d_model//4, 2, 1),(third_layer_channel, configs.d_model//4, 1, 1)]]

        print("layer_parameter_lists:{}".format(layer_parameter_lists))
        single_cnn_list = []
        self.layer_parameter_lists = layer_parameter_lists
        for layer_parameter_list in layer_parameter_lists:
            single_cnn_list.append(MultiConv(layer_parameter_list))
        self.conv_list = nn.ModuleList(single_cnn_list) 

    def forward(self, x):
        B, T, N = x.size()
        # period_list, period_weight = FFT_for_Period(x, self.k)

        X = x
        for layer in self.conv_list:
            # layer.cnns[3].dilution = period_list[0]
            X = layer(X)

        return X


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
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        self.pre_linear = Linear_variableaxis(configs.enc_in, configs.enc_in)

        self.layer = 4
        self.model = nn.ModuleList([DMFCN_Block(configs,True)]+[DMFCN_Block(configs)
                                    for _ in range(self.layer-1)])
        # self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.projection = nn.Linear(
                (configs.d_model//4)*4, configs.num_class)
            self.averagepool = nn.AdaptiveAvgPool1d(1)

        if self.task_name == 'regression':
            self.act = F.gelu
            self.projection = nn.Linear(
                (configs.d_model//4)*4, configs.num_class)
            self.averagepool = nn.AdaptiveAvgPool1d(1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

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
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
        # porject back
        dec_out = self.projection(enc_out)

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

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]\
        enc_out = x_enc
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = enc_out
        # (batch_size, seq_length * d_model)
        # output = output.reshape(output.shape[0], -1)
        output = self.averagepool(output.transpose(1,2)).squeeze(-1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def regression(self, x_enc, x_mark_enc):
        # embedding
        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # enc_out = self.pre_linear(x_enc)
        enc_out = x_enc
        # TimesNet
        for i in range(self.layer):
            # enc_out = self.layer_norm(self.model[i](enc_out))
            enc_out = self.model[i](enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = enc_out
        # (batch_size, seq_length * d_model)
        # output = output.reshape(output.shape[0], -1)
        output = self.averagepool(output.transpose(1,2)).squeeze(-1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

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


class MultiConv(torch.nn.Module):
    def __init__(self, layer_parameter_list, **kwargs):
        super(MultiConv, self).__init__()
        
        cnn_list = []
        all_out_channels = sum(layer_parameter[1] for layer_parameter in layer_parameter_list)
        for layer_parameter in layer_parameter_list:
            input_channels,out_channels,kernel_size,dilation = layer_parameter
            cnn_list.append(torch.nn.Conv1d(in_channels=input_channels, out_channels=out_channels, \
            dilation = dilation, kernel_size=kernel_size, stride=1, padding='same'))
        self.cnns = nn.ModuleList(cnn_list)
        self.bn = torch.nn.BatchNorm1d(num_features= all_out_channels)
        self.activation = nn.ReLU()
        return
    def forward(self, x):
        x = x.transpose(1,2)
        xs = []
        for cnn in self.cnns:
            xs.append(cnn(x))
        h = torch.cat(xs,dim = -2)
        h = self.bn(h)
        h = self.activation(h)
        h = h.transpose(1,2)
        return h