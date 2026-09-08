# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import numpy as np
import scipy.fft as fft
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

# -------------------------- Global Parameters --------------------------
RESULT_ROOT = 'Odour_saturated_results'          # Root directory for saving results
folder_path = r'Odour_saturated_dataset'         # Data folder containing Excel files

# -------------------------- 1. Data Loading (group by prefix, 10 samples per class) --------------------------
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
if not file_list:
    raise FileNotFoundError(f"No Excel files found in {folder_path}")

def extract_class_prefix(file_name):
    # Extract leading number before '_' (class ID: 1-103)
    match = re.match(r'^(\d+)_', file_name)
    if not match:
        raise ValueError(f"File name {file_name} does not start with 'number_' pattern")
    return match.group(1)

class_file_dict = {}
for file_name in file_list:
    prefix = extract_class_prefix(file_name)
    class_file_dict.setdefault(prefix, []).append(file_name)

sorted_class_prefixes = sorted(class_file_dict.keys(), key=int)
n_classes = len(sorted_class_prefixes)
samples_per_class = 10

if n_classes != 103:
    raise ValueError(f"Expected 103 classes, found {n_classes} ({sorted_class_prefixes})")
for prefix in sorted_class_prefixes:
    if len(class_file_dict[prefix]) != samples_per_class:
        raise ValueError(f"Class {prefix} has {len(class_file_dict[prefix])} files, expected 10")

data = []
labels = []
class_label_map = {prefix: idx for idx, prefix in enumerate(sorted_class_prefixes)}

for prefix in sorted_class_prefixes:
    class_files = class_file_dict[prefix]
    class_label = class_label_map[prefix]

    def extract_file_index(file_name):
        # New naming: e.g., "1_1.xlsx" -> extract trailing number
        match = re.search(r'_(\d+)\.xlsx$', file_name)
        return int(match.group(1)) if match else 0

    class_files_sorted = sorted(class_files, key=extract_file_index)

    for file_name in class_files_sorted:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path).iloc[:, 1:]       # skip first column (assumed index)
        data.append(df.values)
        labels.append(class_label)

# Unify sequence length by cropping to the shortest sample
min_rows = min(len(sample) for sample in data)
data = [sample[:min_rows] for sample in data]
data = np.array(data)          # (1030, T, 24)
labels = np.array(labels)      # (1030,)

print("=" * 60)
print("Data loading completed")
print("=" * 60)
print(f"Total samples: {len(data)} (103 classes, 10 each)")
print(f"Sample shape: {data[0].shape} (time steps × channels)")
print(f"Label mapping (first 5): {[(f'Class {p}', f'Label {idx}') for p, idx in list(class_label_map.items())[:5]]} ...")
print(f"Label distribution: {np.bincount(labels)[:10]} ... (each 10)")

# -------------------------- 2. Preprocessing (with FFT fusion and standardization) --------------------------
select_channel = list(range(24))          # use all 24 sensor channels
timeseries_data = data

# Compute FFT magnitude and fuse with time-domain features (weighted)
frequency_data = np.empty_like(timeseries_data)
for i, sample in enumerate(timeseries_data):
    for j, channel in enumerate(sample):
        freq_transform = fft.fft(channel)
        freq_magnitude = np.abs(freq_transform)
        frequency_data[i, j] = freq_magnitude

data = timeseries_data   # keep only time-domain (no fusion in this version)

# Standardize (reshape to 2D, fit scaler, then reshape back)
scaler = StandardScaler()
data_shape = data.shape
data = scaler.fit_transform(
    data.reshape(-1, data_shape[1] * data_shape[2])
).reshape(data_shape)

# -------------------------- 3. PyTorch Data Preparation (10% test per class) --------------------------
seed_value = 41
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
full_dataset = TensorDataset(data_tensor, labels_tensor)

train_indices = []
test_indices = []
test_size_per_class = 1/10   # 1 sample per class for test

for class_label in range(103):
    class_global_indices = np.where(labels == class_label)[0]
    train_idx, test_idx = train_test_split(
        class_global_indices,
        test_size=test_size_per_class,
        random_state=42,
        stratify=[class_label] * len(class_global_indices)
    )
    train_indices.extend(train_idx)
    test_indices.extend(test_idx)

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_labels = [labels[i] for i in train_indices]
test_labels = [labels[i] for i in test_indices]
print("\n" + "=" * 60)
print("Train/Test Split Results")
print("=" * 60)
print(f"Train samples: {len(train_dataset)} (9 per class)")
print(f"Train label distribution (first 10): {np.bincount(train_labels)[:10]} ...")
print(f"Test samples: {len(test_dataset)} (1 per class)")
print(f"Test label distribution (first 10): {np.bincount(test_labels)[:10]} ...")

# -------------------------- 4. CNN-LSTM Model (input 24 channels, output 103 classes) --------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, in_channel=len(select_channel), out_channel=103,
                 lstm_hidden_dim=128, lstm_layers=1):
        super(CNN_LSTM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        with torch.no_grad():
            test_input = torch.randn(1, data_tensor.shape[1], in_channel)
            test_input = test_input.permute(0, 2, 1)
            test_out = self.layer1(test_input)
            test_out = self.layer2(test_out)
            test_out = self.layer3(test_out)
            test_out = test_out.permute(0, 2, 1)
            lstm_out, _ = self.lstm(test_out)
            self.fc_input_dim = lstm_out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.fc_input_dim, 200)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, out_channel)

    def forward(self, x):
        x = x.permute(0, 2, 1)[:, select_channel, :]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(lstm_out.size(0), -1)
        x = self.fc1(lstm_out)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------- 5. Model Training and Evaluation --------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n" + "=" * 60)
print("Model Training Configuration")
print("=" * 60)
print(f"Device: {device}")
print(f"Epochs: 300")
print(f"Batch size: 64")
print(f"Optimizer: Adam (lr=1e-4, weight_decay=1e-5)")
print(f"Loss: CrossEntropyLoss")

model = CNN_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, 'test'), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, 'all'), exist_ok=True)

epochs = 300
best_test_accuracy = 0.0
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

print("\n" + "=" * 60)
print("Training starts (300 epochs)")
print("=" * 60)

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

    train_loss = running_train_loss / total_train
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_targets_test = []
    all_preds_test = []
    all_probs_test = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)

            running_test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()

            all_targets_test.extend(targets.cpu().numpy())
            all_preds_test.extend(predicted.cpu().numpy())
            all_probs_test.extend(probs.cpu().numpy())

    test_loss = running_test_loss / total_test
    test_acc = correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc
        torch.save(model.state_dict(), os.path.join(RESULT_ROOT, 'best_model_params.pth'))
        print(f"Epoch {epoch + 1:3d} | Test accuracy improved to {best_test_accuracy:.4f} | Model saved")
    else:
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# -------------------------- Evaluate best model on test set --------------------------
print("\n" + "=" * 60)
print("Best Model Evaluation on Test Set")
print("=" * 60)

model.load_state_dict(torch.load(os.path.join(RESULT_ROOT, 'best_model_params.pth')))
model.eval()

all_targets_test = []
all_preds_test = []
all_probs_test = []
all_sample_indices_test = []

with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        batch_indices = test_loader.dataset.indices[idx * test_loader.batch_size: (idx + 1) * test_loader.batch_size]
        all_sample_indices_test.extend(batch_indices)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)

        all_targets_test.extend(targets.cpu().numpy())
        all_preds_test.extend(predicted.cpu().numpy())
        all_probs_test.extend(probs.cpu().numpy())

all_targets_test = np.array(all_targets_test)
all_preds_test = np.array(all_preds_test)
all_probs_test = np.array(all_probs_test)
all_sample_indices_test = np.array(all_sample_indices_test)

test_acc = accuracy_score(all_targets_test, all_preds_test)
test_recall = recall_score(all_targets_test, all_preds_test, average='weighted')
test_f1 = f1_score(all_targets_test, all_preds_test, average='weighted')

label_class_map = {idx: prefix for prefix, idx in class_label_map.items()}
all_targets_test_class = [label_class_map[label] for label in all_targets_test]
all_preds_test_class = [label_class_map[label] for label in all_preds_test]

test_metrics = (
    f"Test Set Evaluation Metrics (Best Model)\n"
    f"=====================\n"
    f"Best Test Accuracy: {best_test_accuracy:.4f}\n"
    f"Accuracy: {test_acc:.4f}\n"
    f"Recall (weighted): {test_recall:.4f}\n"
    f"F1-Score (weighted): {test_f1:.4f}\n"
    f"Label mapping: Label 0 → Class 1, ... Label 102 → Class 103"
)
print(test_metrics)
with open(os.path.join(RESULT_ROOT, 'test', 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write(test_metrics)

# Note: predictions_vs_targets.csv has been removed as requested.

# Confusion matrix
plt.figure(figsize=(22, 20))
class_prefixes_sorted = [label_class_map[label] for label in range(103)]
conf_matrix_test = confusion_matrix(
    all_targets_test_class,
    all_preds_test_class,
    labels=class_prefixes_sorted
)
conf_matrix_test_norm = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]

# Save confusion matrix to Excel
conf_matrix_raw_df = pd.DataFrame(
    data=conf_matrix_test,
    index=[f'True_Class_{p}' for p in class_prefixes_sorted],
    columns=[f'Pred_Class_{p}' for p in class_prefixes_sorted]
)
conf_matrix_norm_df = pd.DataFrame(
    data=np.round(conf_matrix_test_norm, 4),
    index=[f'True_Class_{p}' for p in class_prefixes_sorted],
    columns=[f'Pred_Class_{p}' for p in class_prefixes_sorted]
)
with pd.ExcelWriter(
    os.path.join(RESULT_ROOT, 'test', 'confusion_matrix.xlsx'),
    engine='openpyxl'
) as writer:
    conf_matrix_raw_df.to_excel(writer, sheet_name='Raw Counts', index=True)
    conf_matrix_norm_df.to_excel(writer, sheet_name='Normalized', index=True)

# Plot heatmap
sns.heatmap(
    conf_matrix_test_norm,
    annot=conf_matrix_test,
    fmt='d',
    cmap='Blues',
    xticklabels=[f'Class{p}' for p in class_prefixes_sorted],
    yticklabels=[f'Class{p}' for p in class_prefixes_sorted],
    cbar_kws={'label': 'Normalized Prediction Ratio'},
    annot_kws={'fontsize': 3}
)
plt.xlabel('Predicted Class', fontsize=14)
plt.ylabel('True Class', fontsize=14)
plt.title('Test Confusion Matrix (103 classes, Best Model)', fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_ROOT, 'test', 'confusion_matrix.png'), dpi=300)
plt.close()

# Training curves
plt.figure(figsize=(12, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='#1f77b4', linestyle='-', linewidth=2)
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='#ff7f0e', linestyle='--', linewidth=2)
plt.axhline(y=min(test_losses), color='#ff7f0e', linestyle=':', alpha=0.7,
            label=f'Min Test Loss: {min(test_losses):.4f}')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Test Loss Curves (103-class task)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_ROOT, 'all', 'loss_curves.png'), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='#2ca02c', linestyle='-', linewidth=2)
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='#d62728', linestyle='--', linewidth=2)
plt.axhline(y=best_test_accuracy, color='#d62728', linestyle=':', alpha=0.7,
            label=f'Best Test Acc: {best_test_accuracy:.4f}')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and Test Accuracy Curves (103-class task)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_ROOT, 'all', 'accuracy_curves.png'), dpi=300)
plt.close()

train_process_df = pd.DataFrame({
    'Epoch': range(1, epochs + 1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies
})
train_process_df.to_csv(
    os.path.join(RESULT_ROOT, 'all', 'train_process_data.csv'),
    index=False, encoding='utf-8-sig'
)

# PR curves
print("\n" + "=" * 60)
print("Plotting test PR curves (103 classes, showing key classes)")
print("=" * 60)

y_true_bin = label_binarize(all_targets_test, classes=np.arange(103))

precision = dict()
recall = dict()
average_precision = dict()

plt.figure(figsize=(14, 12))
colors = plt.cm.get_cmap('tab20b', 103)

pr_data = {
    'Class Prefix': [],
    'Recall': [],
    'Precision': [],
    'Average Precision (AP)': []
}

ap_list = [(label, average_precision_score(y_true_bin[:, label], all_probs_test[:, label]))
           for label in range(103)]
ap_list_sorted = sorted(ap_list, key=lambda x: x[1], reverse=True)

show_labels = list(range(10)) + list(range(93, 103))
show_labels += [x[0] for x in ap_list_sorted[:5]]
show_labels += [x[0] for x in ap_list_sorted[-5:]]
show_labels = list(set(show_labels))
show_labels.sort()

for label in range(103):
    class_prefix = label_class_map[label]
    precision[label], recall[label], _ = precision_recall_curve(
        y_true_bin[:, label],
        all_probs_test[:, label]
    )
    average_precision[label] = average_precision_score(
        y_true_bin[:, label],
        all_probs_test[:, label]
    )
    pr_data['Class Prefix'].extend([f'Class {class_prefix}'] * len(recall[label]))
    pr_data['Recall'].extend(recall[label])
    pr_data['Precision'].extend(precision[label])
    pr_data['Average Precision (AP)'].extend([average_precision[label]] * len(recall[label]))

    if label in show_labels:
        plt.plot(
            recall[label],
            precision[label],
            color=colors(label),
            lw=2,
            alpha=0.8,
            label=f'Class {class_prefix} (AP={average_precision[label]:.3f})'
        )

precision['micro'], recall['micro'], _ = precision_recall_curve(
    y_true_bin.ravel(),
    all_probs_test.ravel()
)
average_precision['micro'] = average_precision_score(
    y_true_bin, all_probs_test, average='micro'
)
plt.plot(
    recall['micro'],
    precision['micro'],
    color='black',
    linestyle=':',
    linewidth=4,
    label=f'Micro-average (AP={average_precision["micro"]:.3f})'
)
pr_data['Class Prefix'].extend(['Micro-average'] * len(recall['micro']))
pr_data['Recall'].extend(recall['micro'])
pr_data['Precision'].extend(precision['micro'])
pr_data['Average Precision (AP)'].extend([average_precision['micro']] * len(recall['micro']))

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Test PR Curves (103 classes, key classes shown)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_ROOT, 'test', 'pr_curve.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.close()

pr_df = pd.DataFrame(pr_data)
pr_df.to_csv(
    os.path.join(RESULT_ROOT, 'test', 'pr_curve_data.csv'),
    index=False, encoding='utf-8-sig'
)

# -------------------------- 6. Summary --------------------------
print(f"\n" + "=" * 80)
print("All tasks completed! Results saved to:")
print(f"{os.path.abspath(RESULT_ROOT)}")
print("=" * 80)
print("Key result files:")
print("1. test/metrics.txt                - Core test metrics (accuracy/recall/F1)")
print("2. test/confusion_matrix.png       - Confusion matrix heatmap")
print("3. test/confusion_matrix.xlsx      - Confusion matrix (raw + normalized sheets)")
print("4. test/pr_curve.png               - Multi-class PR curves")
print("5. test/pr_curve_data.csv          - PR curve data points")
print("6. all/train_process_data.csv      - Training history")
print("7. all/loss_curves.png             - Loss curves")
print("8. all/accuracy_curves.png         - Accuracy curves")
print("9. best_model_params.pth           - Best model parameters")
print("=" * 80)