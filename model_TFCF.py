import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt
import numpy as np


def Wavelet_for_Period(x, scale=16):
    scales = 2 ** np.linspace(-1, scale, 8)
    coeffs, freqs = ptwt.cwt(x, scales, "morl")
    return coeffs, freqs


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)

class TFCFNet(nn.Module):
    def __init__(self, input_size=36, cnn_channels=96, output_size=3):
        super(TFCFNet, self).__init__()

        self.conv3 = nn.Sequential(nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding='same'),
                                   nn.BatchNorm1d(cnn_channels), nn.ReLU())
        self.conv32 = nn.Sequential(nn.Conv1d(cnn_channels, cnn_channels, kernel_size=32, padding='same'),
                                    nn.BatchNorm1d(cnn_channels), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv1d(input_size, cnn_channels, kernel_size=5, padding='same'),
                                   nn.BatchNorm1d(cnn_channels), nn.ReLU())
        self.conv16 = nn.Sequential(nn.Conv1d(cnn_channels, cnn_channels, kernel_size=16, padding='same'),
                                    nn.BatchNorm1d(cnn_channels), nn.ReLU())

        self.conv7 = nn.Sequential(nn.Conv1d(input_size, cnn_channels, kernel_size=7, padding='same'),
                                   nn.BatchNorm1d(cnn_channels), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv1d(cnn_channels, cnn_channels, kernel_size=8, padding='same'),
                                   nn.BatchNorm1d(cnn_channels), nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv1d(input_size, cnn_channels, kernel_size=1, padding='same'),
                                   nn.BatchNorm1d(cnn_channels), nn.ReLU())


        self.mstfe_reduce = nn.Sequential(
            nn.Conv1d(cnn_channels * 4, cnn_channels, kernel_size=1, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )

        self.dw_cnn = nn.Sequential(
            nn.Conv2d(input_size, cnn_channels, 3, 1, 1, groups=6, bias=False),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cnn_channels * 2), nn.ReLU(),
            SimAM(),

            nn.Conv2d(cnn_channels * 2, cnn_channels * 2, 3, 1, 1, groups=6, bias=False),
            nn.BatchNorm2d(cnn_channels * 2), nn.ReLU(),
            nn.Conv2d(cnn_channels * 2, cnn_channels * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cnn_channels * 4), nn.ReLU(),
            SimAM(),

            nn.Conv2d(cnn_channels * 4, cnn_channels * 4, 3, 1, 1, groups=6, bias=False),
            nn.BatchNorm2d(cnn_channels * 4), nn.ReLU(),
            nn.Conv2d(cnn_channels * 4, cnn_channels * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cnn_channels * 8), nn.ReLU(),
            SimAM(),
        )
        self.scale_conv = nn.Conv2d(cnn_channels * 8, cnn_channels, kernel_size=(8, 1), stride=1, padding=(0, 0),
                                    groups=2)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.gelu = nn.GELU()

        self.t_reshape = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=1)
        self.t_bn = nn.BatchNorm1d(cnn_channels)
        self.f_reshape = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=1)
        self.f_bn = nn.BatchNorm1d(cnn_channels)

        self.fc_alpha = nn.Linear(1, 1)
        self.fc_beta = nn.Linear(1, 1)

        self.t_filter_conv = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=1)
        self.f_filter_conv = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=1)

        self.final_bn = nn.BatchNorm1d(cnn_channels)
        self.final_conv = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=1)

        self.fc = nn.Linear(cnn_channels, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        
        coeffs, _ = Wavelet_for_Period(x, 1)

        coeffs = coeffs.permute(1, 2, 0, 3).float()

        wave = self.dw_cnn(coeffs)  

        f_feat = self.scale_conv(wave).squeeze(2)

        x_1 = self.conv32(self.conv3(x))
        x_2_1 = self.conv5(x)
        x_2 = self.conv16(x_1 + x_2_1)

        x_3_1 = self.conv7(x)
        x_3 = self.conv8(x_3_1 + x_2)

        x_4 = self.conv1(x) + x_3

        t_feat = torch.cat([x_1, x_2, x_3, x_4], dim=1)  
        t_feat = self.mstfe_reduce(t_feat) 

        t_pooled = self.aap(t_feat)
        t_res = self.t_reshape(self.gelu(self.t_bn(t_pooled)))

        f_pooled = self.aap(f_feat)
        f_res = self.f_reshape(self.gelu(self.f_bn(f_pooled)))

        s = F.cosine_similarity(t_res, f_res, dim=1)

        alpha = torch.sigmoid(self.fc_alpha(s)).unsqueeze(-1)
        beta = torch.sigmoid(self.fc_beta(s)).unsqueeze(-1)

        sum_weights = alpha + beta + 1e-8
        alpha = alpha / sum_weights
        beta = beta / sum_weights

        t_filter = torch.sigmoid(self.t_filter_conv(t_res)) * t_feat
        f_filter = torch.sigmoid(self.f_filter_conv(f_res)) * f_feat

        f_fusion = alpha * t_filter + beta * f_filter

        f_final = self.final_conv(self.gelu(self.final_bn(f_fusion))) * f_fusion

        out = self.fc(f_final[:, :, -1])
        return out
