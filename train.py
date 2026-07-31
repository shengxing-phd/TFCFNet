import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from model_TFCF import WaveFFT

from data_loader import CustomDataset, create_inout_sequences, TimeSeriesDataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

model_name = "kanfft"

# Model parameters
input_size = 36
lstm_hidden_size = 128
lstm_num_layers = 3
cnn_channels = 96
output_size = 3  

# Change batch size here
batch_size = 96  # Change this value to your desired batch size
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = WaveFFT(input_size, cnn_channels, output_size)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True, weight_decay=1e-4)

# Paths to training and testing data
# train_csv_path = "data/UAH_driveset.csv"
# train_csv_path = "data/fordTrain.csv"
ext = train_csv_path
Dataset = CustomDataset(train_csv_path)


train_size = int(0.8 * len(Dataset))
test_size = len(Dataset) - train_size
train_data, test_data = torch.utils.data.random_split(Dataset, [train_size, test_size])

train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

epochs = 100
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 96,36
        optimizer.zero_grad()
        outputs = model(inputs)
        # labels = torch.from_numpy(np.array(labels).long()).to(device)
        # loss = criterion(outputs, labels)
        outputs = Tensor.cpu(outputs).float().to(device)
        # print(outputs.shape)
        # outputs = outputs[:, 0, :]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        _, predicted_train = torch.max(outputs.data, dim=1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    epoch_train_loss /= len(train_data_loader)
    train_loss_values.append(epoch_train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracy_values.append(train_accuracy)

    # Testing phase
    model.eval()
    epoch_test_loss = 0.0
    correct_test = 0
    total_test = 0
    recall_test = 0
    acc_test = 0
    i = 0
    f1_test = 0
    pre_test = 0
    # auc_scores = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # labels = torch.from_numpy(np.array(labels).long()).to(device)
            outputs = Tensor.cpu(outputs).float().to(device)
            # outputs = outputs[:, 0, :]
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()

            _, predicted_test = torch.max(outputs.data, 1)
            predicted_test = Tensor.cpu(predicted_test)
            labels = Tensor.cpu(labels)
            total_test += labels.size(0)
            prob_new = torch.nn.functional.softmax(outputs, dim=1)
            correct_test += (predicted_test == labels).sum().item()
            acc_test += accuracy_score(labels, predicted_test).sum().item()
            f1_test += f1_score(labels, predicted_test, average='weighted').sum().item()
            pre_test += precision_score(labels, predicted_test, average='weighted').sum().item()
            recall_test += recall_score(predicted_test, labels, average='weighted').sum().item()
            i += 1

    epoch_test_loss /= len(test_data_loader)
    test_loss_values.append(epoch_test_loss)
    test_accuracy = 100 * correct_test / total_test
    recall_test = 100 * recall_test / i
    acc_test = 100 * acc_test / i
    f1_test = 100 * f1_test / i
    pre_test = 100 * pre_test / i
    test_accuracy_values.append(test_accuracy)

    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy:.2f},Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, '
            f'Test Acc: {acc_test:.2f}%, Test Pre: {pre_test:.2f}%, Test Recall: {recall_test:.2f}%, Test F1: {f1_test:.2f}%')

# Specify output folder for saving the model, CSV file, and plots
output_folder1 = r"Models"  # Replace with desired path
os.makedirs(output_folder1, exist_ok=True)  # Create folder if it doesn't exist
output_folder2 = "CSV"  # Replace with desired path
os.makedirs(output_folder2, exist_ok=True)  # Create folder if it doesn't exist
output_folder3 = "Plots"  # Replace with desired path
os.makedirs(output_folder3, exist_ok=True)  # Create folder if it doesn't exist

train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}.csv")
train_info_df.to_csv(csv_path, index=False)
print(f"Training data saved at {csv_path}")
# Save the model state
model_path = os.path.join(output_folder1, f"{model_name}.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")

train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}.csv")
train_info_df.to_csv(csv_path, index=False)
print(f"Training data saved at {csv_path}")

# Plot the loss and accuracy on the same figure
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')
plt.title('Training and Testing Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_values, label='Testing Accuracy')
plt.title('Training and Testing Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Save plots as PNG and PDF
png_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
print(f"Plots saved at {png_file_path} and {pdf_file_path}")
plt.tight_layout()
plt.show()
