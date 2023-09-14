import torch
import numpy as np
import torch.nn as nn
from model.conv_model import conv_model
from model.transformer import Transformer


class MSTransformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, d_model, n_heads, e_layers, d_layers,
                 dff, query_size, value_size, attention_size=None, dropout=0.3, chunk_mode=None,
                 pe='original', pe_period=None, device=torch.device('cuda:0'),
                 kernel_size_t=3, kernel_size_d=3, conv_layers=2, sampling_rate=6):

        super(MSTransformer, self).__init__()

        # 堆叠多层粗粒度卷积补全模型
        self.layers_conv = nn.ModuleList([conv_model(
                                        kernel_size=[kernel_size_t, kernel_size_d])
                                        for _ in range(conv_layers)])

        self.msTransformer = Transformer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            d_model=d_model,
            dff=dff,
            q=query_size,
            v=value_size,
            h=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            attention_size=attention_size,
            dropout=dropout,
            chunk_mode=chunk_mode,
            pe=pe,
            pe_period=pe_period,
            sampling_rate=sampling_rate)

    def forward(self, x, y, mask_map, pred_mask_map, enc_types, pred_types, IFALL=1):
        # x:[batch_size, K, d_input]含有缺失值
        # y:[batch_size, Y, d_y]含有缺失值
        # mask_map: 数据缺失为1，不缺失为0
        conv_x = x
        # print("原始输入数据:", original_x)
        # 进行粗粒度补全
        for layer in self.layers_conv:
            conv_x = layer(conv_x, mask_map)
        # 对已经采样点的数值采用采样的真实数据
        # print(x)
        res = conv_x * mask_map
        # print("补全的缺失数据:", res)
        coarse_x = x + res
        # print("coarse_x", coarse_x)
        # 如果是全局的话，这里可以考虑给decoder的输入也一样的补全
        if IFALL:
            decoding = y[:, :x.shape[1], :]
            for layer in self.layers_conv:
                decoding = layer(decoding, mask_map)
            res_y = decoding * mask_map
            y[:, :x.shape[1], :] = res_y

        output = self.msTransformer(coarse_x, y, mask_map, pred_mask_map, enc_types, pred_types, IFALL)
        return output


