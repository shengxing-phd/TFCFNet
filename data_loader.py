import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(Tensor.cpu(sequence).numpy()), torch.Tensor(Tensor.cpu(label).numpy())


def create_inout_sequences(input_data, tw, pre_len):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class CustomDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.inputs = self.data.iloc[:, :-1].values
        # self.inputs = self.data.iloc[:, :-1].astype(float).values
        self.labels = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_tensor, label_tensor
