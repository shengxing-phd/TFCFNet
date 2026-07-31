import torch
import torch.nn as nn
# from thop import profile
# import torchstat
# import torchsummary
# from thop import profile
# from TCN import TemporalConvNet
# from FANLayer import FANLayer
# from KAN import KAN
# from StarNet import StarNet
import ptwt
import numpy as np


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=1, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def Wavelet_for_Period(x, scale=16):
    scales = 2 ** np.linspace(-1, scale, 8)
    coeffs, freqs = ptwt.cwt(x, scales, "morl")
    return coeffs, freqs


# 无参注意力
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class WaveFFT(nn.Module):
    def __init__(self, input_size=36, cnn_channels=96, output_size=3):
        # def __init__(self, input_size, cnn_channels, output_size):
        super(WaveFFT, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=7, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=1, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.cnn1_1 = nn.Sequential(
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=32, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.cnn2_1 = nn.Sequential(
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=16, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.cnn3_1 = nn.Sequential(
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=8, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.adapool = nn.AdaptiveMaxPool1d(3)
        self.conv2d = nn.Sequential(
            nn.Conv2d(input_size, cnn_channels, 3, 1, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(cnn_channels, cnn_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(cnn_channels * 2, cnn_channels * 4, 3, 1, 1),
            nn.BatchNorm2d(cnn_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(input_size, cnn_channels, 3, 1, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(cnn_channels * 4, cnn_channels * 8, 3, 1, 1),
            nn.BatchNorm2d(cnn_channels * 8),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False),
        )
        self.dw_cnn = nn.Sequential(
            # dw
            nn.Conv2d(input_size, cnn_channels, 3, 1, 1, groups=6, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            # pw
            nn.Conv2d(cnn_channels, cnn_channels * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            # simam
            SimAM(),
            # dw
            nn.Conv2d(cnn_channels * 2, cnn_channels * 2, 3, 1, 1, groups=6, bias=False),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            # pw
            nn.Conv2d(cnn_channels * 2, cnn_channels * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cnn_channels * 4),
            nn.ReLU(),
            # simam
            SimAM(),
            # dw
            nn.Conv2d(cnn_channels * 4, cnn_channels * 4, 3, 1, 1, groups=6, bias=False),
            nn.BatchNorm2d(cnn_channels * 4),
            nn.ReLU(),
            # pw
            nn.Conv2d(cnn_channels * 4, cnn_channels * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cnn_channels * 8),
            nn.ReLU(),
            # simam
            SimAM(),
        )
        self.scale_conv = nn.Conv2d(
            in_channels=cnn_channels * 8,
            out_channels=cnn_channels,
            kernel_size=(8, 1),
            stride=1,
            padding=(0, 0),
            groups=2)
        # self.complex_weight = nn.Parameter(torch.randn(input_size, 2, dtype=torch.float32) * 0.02)
        # self.starnet = StarNet(base_dim=input_size, num_classes=output_size)
        # self.tcn = TemporalConvNet(input_size, [1, 2, 4])
        self.fc = nn.Linear(cnn_channels, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D, but got {x.dim()}D tensor instead.")
        # 小波变换分支
        x_wave = x
        coeffs = Wavelet_for_Period(x_wave.permute(0, 2, 1), 1)[0].permute(1, 2, 0, 3).float()  # 96,36,8,1
        # wavelet_res = self.period_conv(coeffs)  # 96,36,8,1
        # wavelet_res = self.scale_conv(wavelet_res).squeeze(2).permute(0, 2, 1)  # 96,1,96
        wave = self.dw_cnn(coeffs)  # 96,768,8,1
        wavelet_res = self.scale_conv(wave).squeeze(2).permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x1 = self.cnn4(x)
        x2 = self.cnn1(x)
        x3 = self.cnn2(x)
        x4 = self.cnn3(x)
        x2_2 = self.cnn1_1(x2)
        x3_1 = x2_2 + x3
        x3_3 = self.cnn2_1(x3_1)
        x4_1 = x3_3 + x4
        x4_4 = self.cnn3_1(x4_1)
        x4_4 = x4_4 + x1  # 96,96,1

        x_out = torch.cat([x2_2 + x3_3 + x4_4], dim=1)
        # print(x_out.shape)
        # print(wavelet_res.shape)
        # x = (1 - 0.86 ** 10) * wavelet_res + (0.86 ** 10) * x_cnn
        # x = 0.78 * wavelet_res + 0.22 * x_out
        x = 0.8 * wavelet_res + 0.2 * x_out

        x = self.fc(x[:, -1, :])
        return x


if __name__ == "__main__":
    model = WaveFFT()
    # torchstat.stat(model, (96, 36))
    # torchsummary.summary(model, input_size=[(96, 36)], device="cpu")
    flops, params = profile(model, inputs=(torch.randn(1, 96, 36)))
    print('Flops: % .4fG' % (flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))
