# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import numpy as np
import scipy.fft as fft
import pywt  # kept for compatibility
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------- Global Parameters (removed FFT, wavelet, gradient weights) --------------------------
DATA_FOLDER_PATH = r'Single-gas_dataset'  # data folder path (keep original Chinese name as per user)
RESULT_ROOT = 'Single-gas_classification_results'  # root directory for results (English)

# Data processing parameters
SELECT_CHANNEL = [0, 1, 2, 3, 4, 5, 6, 7]  # selected channel indices
TEST_SIZE_PER_CLASS = 0.1  # test set proportion per class

# Model training parameters
BATCH_SIZE = 64
EPOCHS = 10000
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.3
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 1

# Random seed for reproducibility
SEED_VALUE = 41

# Visualization parameters
PLOT_DPI = 300
SANKY_MIN_VALUE = 0.01

# Set Chinese font for plots (still needed for any Chinese characters in data labels, but we'll use English labels)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. Data Loading --------------------------
folder_path = DATA_FOLDER_PATH

file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
if not file_list:
    raise FileNotFoundError(f"No Excel files found in {folder_path}")

def extract_class_prefix(file_name):
    match = re.match(r'^(\d+)-', file_name)
    if not match:
        raise ValueError(f"File name {file_name} does not start with '数字-' pattern")
    return match.group(1)

class_file_dict = {}
for file_name in file_list:
    class_prefix = extract_class_prefix(file_name)
    class_file_dict.setdefault(class_prefix, []).append(file_name)

sorted_class_prefixes = sorted(class_file_dict.keys(), key=int)
n_classes = len(sorted_class_prefixes)
samples_per_class = 20

if n_classes != 10:
    raise ValueError(f"Expected 10 classes, got {n_classes} ({sorted_class_prefixes})")
for class_prefix in sorted_class_prefixes:
    if len(class_file_dict[class_prefix]) != samples_per_class:
        raise ValueError(f"Class {class_prefix} has {len(class_file_dict[class_prefix])} files, expected 20")

data = []
labels = []
class_label_map = {prefix: idx for idx, prefix in enumerate(sorted_class_prefixes)}

for class_prefix in sorted_class_prefixes:
    class_files = class_file_dict[class_prefix]
    class_label = class_label_map[class_prefix]

    def extract_file_index(file_name):
        # New naming: e.g., "1-0.5,1,2,4,8_1.xlsx" -> extract the trailing number after underscore
        match = re.search(r'_(\d+)\.xlsx$', file_name)
        return int(match.group(1)) if match else 0

    class_files_sorted = sorted(class_files, key=extract_file_index)

    for file_name in class_files_sorted:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path).iloc[:, 1:]  # skip first column (assumed index)
        data.append(df.values)
        labels.append(class_label)

min_rows = min(len(sample) for sample in data)
data = [sample[:min_rows] for sample in data]
data = np.array(data)
labels = np.array(labels)

print("=" * 60)
print("Data loading completed")
print("=" * 60)
print(f"Total samples: {len(data)} (10 classes, 20 each)")
print(f"Sample shape: {data[0].shape} (time steps × channels)")
print(f"Label mapping: {[(f'Class {prefix}', f'Label {idx}') for prefix, idx in class_label_map.items()]}")
print(f"Label distribution: {np.bincount(labels)}")

# -------------------------- 2. Preprocessing (use raw data only, no feature fusion) --------------------------
select_channel = SELECT_CHANNEL
n_channels = len(select_channel)
timeseries_data = data
data = timeseries_data  # directly use raw time series

# Create result folder structure
result_root = RESULT_ROOT
os.makedirs(result_root, exist_ok=True)

# === MODIFIED: removed channel_class_correlation_analysis folder creation ===

# New folder for ablation analysis
model_ablation_path = os.path.join(result_root, 'model_correlation_channel_ablation')
os.makedirs(model_ablation_path, exist_ok=True)

print("\nResult saving paths:")
print(f"Model ablation results path: {model_ablation_path}")
print(f"Path exists: {os.path.exists(model_ablation_path)}")
print(f"Path writable: {os.access(model_ablation_path, os.W_OK)}")

# === MODIFIED: removed original-feature channel-class correlation analysis ===

# Global standardization for model training
scaler = StandardScaler()
data_shape = data.shape
data = scaler.fit_transform(
    data.reshape(-1, data_shape[1] * data_shape[2])
).reshape(data_shape)

# -------------------------- 3. PyTorch Data Preparation --------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split

seed_value = SEED_VALUE
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
test_size_per_class = TEST_SIZE_PER_CLASS

for class_label in range(10):
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_labels = [labels[i] for i in train_indices]
test_labels = [labels[i] for i in test_indices]
print("\n" + "=" * 60)
print("Train/Test Split Results")
print("=" * 60)
print(f"Training samples: {len(train_dataset)} (18 per class)")
print(f"Train label distribution: {np.bincount(train_labels)}")
print(f"Test samples: {len(test_dataset)} (2 per class)")
print(f"Test label distribution: {np.bincount(test_labels)}")

# -------------------------- 4. CNN-LSTM Model Definition --------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, in_channel=len(select_channel), out_channel=10,
                 lstm_hidden_dim=LSTM_HIDDEN_DIM, lstm_layers=LSTM_LAYERS):
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

        self.fc1 = nn.Linear(self.fc_input_dim, 100)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(100, out_channel)

    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)[:, select_channel, :]
        x = self.layer1(x)
        x = self.layer2(x)
        cnn_features = self.layer3(x)
        x = cnn_features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(lstm_out.size(0), -1)
        x = self.fc1(lstm_out)
        x = self.dropout(x)
        x = self.fc2(x)

        if return_features:
            return x, cnn_features
        return x

# -------------------------- 5. Model Training and Evaluation --------------------------
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, average_precision_score)
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n" + "=" * 60)
print("Model Training Configuration")
print("=" * 60)
print(f"Device: {device}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Optimizer: Adam (lr={LEARNING_RATE})")

model = CNN_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

os.makedirs(os.path.join(result_root, 'test'), exist_ok=True)
os.makedirs(os.path.join(result_root, 'all'), exist_ok=True)

epochs = EPOCHS
best_test_accuracy = 0.0
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

print("\n" + "=" * 60)
print(f"Starting training for {EPOCHS} epochs")
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
        torch.save(model.state_dict(), os.path.join(result_root, 'best_model_params.pth'))
        print(f"Epoch {epoch + 1:3d} | Test accuracy improved to {best_test_accuracy:.4f} | Model saved")
    else:
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

print("\n" + "=" * 60)
print("Best Model Evaluation on Test Set")
print("=" * 60)

model.load_state_dict(torch.load(os.path.join(result_root, 'best_model_params.pth')))
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
    f"Test Set Evaluation Metrics (Best Model, Original Features Only)\n"
    f"=====================\n"
    f"Best Test Accuracy: {best_test_accuracy:.4f}\n"
    f"Accuracy: {test_acc:.4f}\n"
    f"Recall (weighted): {test_recall:.4f}\n"
    f"F1-Score (weighted): {test_f1:.4f}\n"
    f"Label mapping: Label 0 → Class 1, ... Label 9 → Class 10"
)
print(test_metrics)
with open(os.path.join(result_root, 'test', 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write(test_metrics)

# === MODIFIED: removed creation of predictions_vs_targets.csv ===

# Confusion matrix
plt.figure(figsize=(12, 10))
class_prefixes_sorted = [label_class_map[label] for label in range(10)]
conf_matrix_test = confusion_matrix(
    all_targets_test_class,
    all_preds_test_class,
    labels=class_prefixes_sorted
)
conf_matrix_test_norm = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]

plt.imshow(conf_matrix_test_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Test Confusion Matrix (Best Model, Original Features Only)', fontsize=14)
plt.colorbar(label='Normalized Prediction Ratio')
tick_marks = np.arange(len(class_prefixes_sorted))
plt.xticks(tick_marks, [f'Class {cp}' for cp in class_prefixes_sorted], rotation=45)
plt.yticks(tick_marks, [f'Class {cp}' for cp in class_prefixes_sorted])

thresh = conf_matrix_test_norm.max() / 2.
for i in range(conf_matrix_test_norm.shape[0]):
    for j in range(conf_matrix_test_norm.shape[1]):
        plt.text(j, i, f"{conf_matrix_test[i, j]}",
                 horizontalalignment="center",
                 color="white" if conf_matrix_test_norm[i, j] > thresh else "black")

plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(result_root, 'test', 'confusion_matrix.png'), dpi=PLOT_DPI)
plt.close()

# Training curves
plt.figure(figsize=(12, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='#1f77b4', linestyle='-', linewidth=2)
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='#ff7f0e', linestyle='--', linewidth=2)
plt.axhline(y=min(test_losses), color='#ff7f0e', linestyle=':', alpha=0.7,
            label=f'Min Test Loss: {min(test_losses):.4f}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves (Original Features Only)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(result_root, 'all', 'loss_curves.png'), dpi=PLOT_DPI)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='#2ca02c', linestyle='-', linewidth=2)
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='#d62728', linestyle='--', linewidth=2)
plt.axhline(y=best_test_accuracy, color='#d62728', linestyle=':', alpha=0.7,
            label=f'Best Test Acc: {best_test_accuracy:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Curves (Original Features Only)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(result_root, 'all', 'accuracy_curves.png'), dpi=PLOT_DPI)
plt.close()

train_process_df = pd.DataFrame({
    'Epoch': range(1, epochs + 1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies
})
train_process_df.to_csv(
    os.path.join(result_root, 'all', 'train_process_data.csv'),
    index=False, encoding='utf-8-sig'
)

# PR Curves
print("\n" + "=" * 60)
print("Plotting test PR curves")
print("=" * 60)

y_true_bin = label_binarize(all_targets_test, classes=np.arange(10))

precision = dict()
recall = dict()
average_precision = dict()

plt.figure(figsize=(12, 10))
colors = plt.cm.get_cmap('tab10', 10)

pr_data = {
    'Class Prefix': [],
    'Recall': [],
    'Precision': [],
    'Average Precision (AP)': []
}

for label in range(10):
    class_prefix = label_class_map[label]
    precision[label], recall[label], _ = precision_recall_curve(
        y_true_bin[:, label],
        all_probs_test[:, label]
    )
    average_precision[label] = average_precision_score(
        y_true_bin[:, label],
        all_probs_test[:, label]
    )
    plt.plot(
        recall[label],
        precision[label],
        color=colors(label),
        lw=2,
        alpha=0.8,
        label=f'Class {class_prefix} (AP={average_precision[label]:.3f})'
    )
    pr_data['Class Prefix'].extend([f'Class {class_prefix}'] * len(recall[label]))
    pr_data['Recall'].extend(recall[label])
    pr_data['Precision'].extend(precision[label])
    pr_data['Average Precision (AP)'].extend([average_precision[label]] * len(recall[label]))

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

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Test Multi-class PR Curves (Best Model, Original Features Only)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(result_root, 'test', 'pr_curve.png'),
    dpi=PLOT_DPI,
    bbox_inches='tight'
)
plt.close()

pr_df = pd.DataFrame(pr_data)
pr_df.to_csv(
    os.path.join(result_root, 'test', 'pr_curve_data.csv'),
    index=False, encoding='utf-8-sig'
)

# -------------------------- 6. Model-based Channel-Class Correlation Analysis (Ablation) --------------------------
print("\n" + "=" * 60)
print("Generating model-based channel-class correlation analysis...")
print("=" * 60)

def get_channel_importance_by_ablation(model, data_tensor, labels, select_channel, device):
    """Channel ablation: zero out a channel, accuracy drop = importance"""
    model.eval()
    n_channels = len(select_channel)
    base_acc = 0.0
    channel_importance = np.zeros(n_channels)

    with torch.no_grad():
        outputs = model(data_tensor.to(device))
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        base_acc = accuracy_score(labels, pred)

        for ch_idx in range(n_channels):
            data_copy = data_tensor.clone()
            data_copy[:, :, select_channel[ch_idx]] = 0.0
            outputs = model(data_copy.to(device))
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            acc = accuracy_score(labels, pred)
            channel_importance[ch_idx] = base_acc - acc

    return channel_importance, base_acc

def compute_channel_class_importance_by_ablation(model, data_tensor, labels, select_channel, device, n_classes=10):
    """Compute importance of each channel for each class via ablation (probability drop)"""
    model.eval()
    n_channels = len(select_channel)
    importance = np.zeros((n_channels, n_classes))

    with torch.no_grad():
        base_probs = []
        for cls in range(n_classes):
            cls_mask = labels == cls
            cls_data = data_tensor[cls_mask].to(device)
            outputs = model(cls_data)
            probs = torch.softmax(outputs, dim=1)[:, cls].mean().item()
            base_probs.append(probs)

        for ch_idx in range(n_channels):
            for cls in range(n_classes):
                cls_mask = labels == cls
                cls_data = data_tensor[cls_mask].clone()
                cls_data[:, :, select_channel[ch_idx]] = 0.0
                outputs = model(cls_data.to(device))
                masked_probs = torch.softmax(outputs, dim=1)[:, cls].mean().item()
                importance[ch_idx, cls] = max(0.0, base_probs[cls] - masked_probs)

    return importance

channel_importance, base_acc = get_channel_importance_by_ablation(
    model, data_tensor, labels, SELECT_CHANNEL, device
)
print(f"Baseline accuracy (all channels): {base_acc:.4f}")
print(f"Channel importance (accuracy drop): {channel_importance}")

model_channel_class_importance = compute_channel_class_importance_by_ablation(
    model, data_tensor, labels, SELECT_CHANNEL, device, n_classes=10
)
print("Channel-class importance matrix shape:", model_channel_class_importance.shape)

# ---- Visualization ----
min_val = np.min(model_channel_class_importance)
max_val = np.max(model_channel_class_importance)
normalized_importance = (model_channel_class_importance - min_val) / (max_val - min_val + 1e-8)

nodes = [f'Channel {i}' for i in range(len(SELECT_CHANNEL))]
nodes += [f'Class {i+1}' for i in range(10)]
node_colors = ['#3498db'] * 8 + ['#e74c3c'] * 10

links = []
for ch in range(8):
    for cls in range(10):
        val = normalized_importance[ch, cls]
        if val > SANKY_MIN_VALUE:
            links.append({
                'source': ch,
                'target': 8 + cls,
                'value': val,
                'color': f'rgba(150, 150, 150, {min(val * 0.8 + 0.2, 1.0)})'
            })

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color=node_colors
    ),
    link=dict(
        source=[l['source'] for l in links],
        target=[l['target'] for l in links],
        value=[l['value'] for l in links],
        color=[l['color'] for l in links]
    )
)])
fig.update_layout(
    title_text="Model-driven Channel-Class Importance Sankey (Ablation)",
    font=dict(family="SimHei, Microsoft YaHei", size=10),
    width=1000,
    height=700,
    margin=dict(l=20, r=20, t=50, b=20)
)
sankey_html_path = os.path.join(model_ablation_path, "model_driven_channel_class_importance_sankey.html")
fig.write_html(sankey_html_path, include_plotlyjs='cdn')
print(f"Sankey saved to: {sankey_html_path}")

# Heatmap
plt.figure(figsize=(12, 8))
im = plt.imshow(model_channel_class_importance, cmap='viridis', aspect='auto')
plt.colorbar(im, label='Importance (Probability Drop)')
plt.title('Channel-Class Importance Heatmap (Ablation)', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Channel Index')
plt.xticks(ticks=np.arange(10), labels=[f'Class {i+1}' for i in range(10)])
plt.yticks(ticks=np.arange(8), labels=[f'Channel {i}' for i in range(8)])
for i in range(8):
    for j in range(10):
        plt.text(j, i, f'{model_channel_class_importance[i, j]:.3f}',
                 ha='center', va='center', color='white' if model_channel_class_importance[i, j] > 0.5 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(model_ablation_path, "model_driven_importance_heatmap.png"), dpi=PLOT_DPI)
plt.close()

# === MODIFIED: removed average channel importance bar chart ===

importance_df = pd.DataFrame(
    model_channel_class_importance,
    index=[f'Channel {i}' for i in range(8)],
    columns=[f'Class {i+1}' for i in range(10)]
)
importance_df.to_csv(os.path.join(model_ablation_path, "model_driven_channel_class_importance_matrix.csv"), encoding='utf-8-sig')
print("Model-driven importance matrix saved.")

# === MODIFIED: removed channel importance comparison (plot and CSV) ===

# -------------------------- 7. Summary --------------------------
print(f"\n" + "=" * 80)
print("All tasks completed! Results saved to:")
print(f"{os.path.abspath(result_root)}")
print("=" * 80)
print("Key result files:")
print("1. model_correlation_channel_ablation/ - Model ablation analysis (Sankey, heatmap, importance matrix)")
print("2. test/metrics.txt - Test set core metrics (original features only)")
print("3. test/confusion_matrix.png - Confusion matrix")
print("4. test/pr_curve.png - Multi-class PR curves")
print("5. all/train_process_data.csv - Training process data")
print("6. all/loss_curves.png - Loss curves")
print("7. all/accuracy_curves.png - Accuracy curves")
print("8. best_model_params.pth - Best model parameters (input: original features)")
print("=" * 80)