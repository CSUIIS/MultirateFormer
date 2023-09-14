import torch.nn as nn
import torch.nn.functional as F
import torch


class conv_model(nn.Module):
    def __init__(self, kernel_size):
        # kernel_size: [time_step, dimention_step]
        super(conv_model, self).__init__()

        self._time_step = kernel_size[0]
        self._dimention_step = kernel_size[1]

        self._conv2d = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=(self._time_step, self._dimention_step))
        self.weight_init()

    def forward(self, x, mask_map):
        # x:[batch_size, K, d_input]
        # mask_map: 数据缺失为1，不缺失为0
        K = x.shape[1]
        original = x
        # print("original:", original)

        # 先使用pad给原始输入边缘补0保证最后输出的形状大小不变
        # print("未变换之前的大小：", x.shape)
        x = x.unsqueeze(1)  # [batch_size, 1, K, d_input]
        # print("变换之后的大小：", x.shape)
        x = F.pad(x, (self._dimention_step - 1, 0, self._time_step - 1, 0))

        # 卷积提取时空特征
        x = self._conv2d(x)
        x = x.squeeze(1)  # [batch_size, K, d_input]
        # print("变换之后的大小：", x.shape)

        # 为了堆叠提取更大范围内的值用于填充缺失值，这个函数里暂时不对原始值还原
        # # 只对缺失值进行填充，真实值不处理
        # mask_map_inverse = 1 - mask_map     # 取反原始屏蔽矩阵
        # res = x * mask_map
        # print("res:", res)
        #
        # # 数据矩阵等于原始值加上填充值
        # x = original + res
        return x

    def weight_init(self):
        # 粗粒度补全的卷积网络参数初始化是取附近几个点的均值
        mean_init = 1 / (self._time_step * self._dimention_step)
        bias_init = 0 / (self._time_step * self._dimention_step)
        torch.nn.init.constant_(self._conv2d.weight, mean_init)
        torch.nn.init.constant_(self._conv2d.bias, bias_init)


