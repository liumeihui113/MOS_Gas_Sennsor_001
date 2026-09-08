import pandas as pd
import os
import re
import numpy as np
import scipy.fft as fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== 【Mode Switch】 ========================
# Mode 1: 'shape'   -> use only waveform shape (per-sample Z-score normalization)
# Mode 2: 'amplitude' -> use only signal amplitude (temporal average, remove waveform)
MODE = 'shape'  # options: 'shape' or 'amplitude'
# ================================================================

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. Data Loading (train and test split) --------------------------
train_folder = r'Odour_saturated_dataset'      # contains 100 classes, each with 10 samples (1_1.xlsx ~ 1_10.xlsx)
test_folder = r'Odour_unsaturated_dataset'     # contains 100 classes, each with 1 sample (1.xlsx ~ 100.xlsx)

# Result root directory
RESULT_ROOT = 'Odour_unsaturated_shap_results'

def extract_class_prefix(file_name):
    """
    Extract class number from filename.
    Supports two formats:
      - train: "1_1.xlsx" -> returns "1"
      - test: "1.xlsx" -> returns "1"
    """
    match = re.match(r'^(\d+)_', file_name)  # train format
    if match:
        return match.group(1)
    match = re.match(r'^(\d+)\.xlsx$', file_name)  # test format
    if match:
        return match.group(1)
    raise ValueError(f"File name {file_name} does not match expected pattern")

# ---------- Load training data (all 100 classes, 10 samples each) ----------
train_files = [f for f in os.listdir(train_folder) if f.endswith('.xlsx')]
if not train_files:
    raise FileNotFoundError(f"No Excel files found in {train_folder}")

train_class_files = {}
for f in train_files:
    cls = extract_class_prefix(f)
    train_class_files.setdefault(cls, []).append(f)

# Keep only classes 1-100 (ignore 101-103 if present)
sorted_class_prefixes = sorted([c for c in train_class_files.keys() if int(c) <= 100], key=int)
n_classes = len(sorted_class_prefixes)
samples_per_class_train = 10

if n_classes != 100:
    raise ValueError(f"Training set has {n_classes} classes, expected 100")
for cls in sorted_class_prefixes:
    if len(train_class_files[cls]) != samples_per_class_train:
        raise ValueError(f"Class {cls} has {len(train_class_files[cls])} files, expected 10")

class_label_map = {prefix: idx for idx, prefix in enumerate(sorted_class_prefixes)}

data_train_raw = []
labels_train = []
for cls in sorted_class_prefixes:
    files = train_class_files[cls]
    # Sort by index (e.g., 1_1.xlsx, 1_2.xlsx, ...)
    def extract_index(fname):
        match = re.search(r'_(\d+)\.xlsx$', fname)   # train format
        return int(match.group(1)) if match else 0
    files_sorted = sorted(files, key=extract_index)
    for fname in files_sorted:
        path = os.path.join(train_folder, fname)
        df = pd.read_excel(path).iloc[:, 1:]  # skip first column (assumed index)
        data_train_raw.append(df.values)
        labels_train.append(class_label_map[cls])

# ---------- Load test data (100 classes, 1 sample each) ----------
test_files = [f for f in os.listdir(test_folder) if f.endswith('.xlsx')]
if not test_files:
    raise FileNotFoundError(f"No Excel files found in {test_folder}")

# Build dict and verify classes match training
test_class_files = {}
for f in test_files:
    cls = extract_class_prefix(f)
    test_class_files.setdefault(cls, []).append(f)

# Check that every training class exists in test set
for cls in sorted_class_prefixes:
    if cls not in test_class_files:
        raise ValueError(f"Test folder missing class {cls}")

# Load test data (each class should have exactly 1 file, but we take the first if multiple)
data_test_raw = []
labels_test = []
for cls in sorted_class_prefixes:
    files = test_class_files[cls]
    # There should be only one file per class, but take the first if more
    fname = files[0] if files else None
    if fname is None:
        raise ValueError(f"Class {cls} has no test file")
    path = os.path.join(test_folder, fname)
    df = pd.read_excel(path).iloc[:, 1:]
    data_test_raw.append(df.values)
    labels_test.append(class_label_map[cls])

# ---------- Unify time steps (crop to the shortest in training set) ----------
min_rows_train = min(len(s) for s in data_train_raw)
data_train = [s[:min_rows_train] for s in data_train_raw]
data_test = [s[:min_rows_train] for s in data_test_raw]   # assume test samples are long enough

data_train = np.array(data_train)   # [1000, T, 24]
labels_train = np.array(labels_train)
data_test = np.array(data_test)     # [100, T, 24]
labels_test = np.array(labels_test)

print("=" * 60)
print(f"Training samples: {len(data_train)}, each shape: {data_train[0].shape}")
print(f"Test samples: {len(data_test)}, each shape: {data_test[0].shape}")
print("=" * 60)

# -------------------------- 2. Core Preprocessing (based on MODE) --------------------------
print(f"\nCurrent mode: {MODE.upper()}")
if MODE == 'shape':
    print("Preprocessing: per-sample Z-score normalization -> keep waveform shape, remove amplitude")
    # Training set
    processed_train = np.zeros_like(data_train)
    for i in range(data_train.shape[0]):
        sample = data_train[i]
        flat = sample.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        if std < 1e-12:
            std = 1.0
        processed_train[i] = (sample - mean) / std
    data_train = processed_train
    # Test set
    processed_test = np.zeros_like(data_test)
    for i in range(data_test.shape[0]):
        sample = data_test[i]
        flat = sample.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        if std < 1e-12:
            std = 1.0
        processed_test[i] = (sample - mean) / std
    data_test = processed_test

elif MODE == 'amplitude':
    print("Preprocessing: temporal average and tile -> keep only signal amplitude, remove waveform dynamics")
    # Training set
    processed_train = np.zeros_like(data_train)
    for i in range(data_train.shape[0]):
        sample = data_train[i]
        channel_means = np.mean(sample, axis=0)
        processed_train[i] = np.tile(channel_means, (data_train.shape[1], 1))
    data_train = processed_train
    # Test set
    processed_test = np.zeros_like(data_test)
    for i in range(data_test.shape[0]):
        sample = data_test[i]
        channel_means = np.mean(sample, axis=0)
        processed_test[i] = np.tile(channel_means, (data_test.shape[1], 1))
    data_test = processed_test

else:
    raise ValueError("MODE must be 'shape' or 'amplitude'")

print(f"Processed train shape: {data_train.shape}, mean: {np.mean(data_train):.4f}, std: {np.std(data_train):.4f}")
print(f"Processed test shape: {data_test.shape}, mean: {np.mean(data_test):.4f}, std: {np.std(data_test):.4f}")

# -------------------------- 3. PyTorch Data Preparation --------------------------
seed_value = 41
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset = TensorDataset(torch.tensor(data_train, dtype=torch.float32),
                              torch.tensor(labels_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(data_test, dtype=torch.float32),
                             torch.tensor(labels_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

# -------------------------- 4. CNN-LSTM Model Definition --------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, in_channel=24, out_channel=100,
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
        # Compute FC input dimension dynamically
        with torch.no_grad():
            test_input = torch.randn(1, data_train.shape[1], in_channel)
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
        x = x.permute(0, 2, 1)
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

# -------------------------- 5. Training and Evaluation --------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create result directory (separated by mode)
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, 'test'), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, 'all'), exist_ok=True)

model = CNN_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

epochs = 300
best_test_accuracy = 0.0
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

print("\nStarting training...")
for epoch in range(epochs):
    # Training
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

    # Testing
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

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# Load best model for final evaluation
model.load_state_dict(torch.load(os.path.join(RESULT_ROOT, 'best_model_params.pth')))
model.eval()

all_targets_test = []
all_preds_test = []
all_probs_test = []
with torch.no_grad():
    for inputs, targets in test_loader:
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

test_acc = accuracy_score(all_targets_test, all_preds_test)
test_recall = recall_score(all_targets_test, all_preds_test, average='weighted')
test_f1 = f1_score(all_targets_test, all_preds_test, average='weighted')

print("\n" + "=" * 60)
print(f"【{MODE.upper()} Mode】Final Test Set Metrics (unsaturated data)")
print(f"Accuracy: {test_acc:.4f}")
print(f"Recall (weighted): {test_recall:.4f}")
print(f"F1-Score (weighted): {test_f1:.4f}")
print("=" * 60)

# Save metrics
with open(os.path.join(RESULT_ROOT, 'test', 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Experiment Mode: {MODE}\n")
    f.write(f"Test set source: {test_folder}\n")
    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
    f.write(f"F1: {test_f1:.4f}\n")

# Confusion matrix
label_class_map = {idx: prefix for prefix, idx in class_label_map.items()}
class_prefixes_sorted = [label_class_map[label] for label in range(100)]
conf_matrix = confusion_matrix(all_targets_test, all_preds_test, labels=range(100))
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(20, 18))
sns.heatmap(conf_matrix_norm, annot=conf_matrix, fmt='d', cmap='Blues',
            xticklabels=[f'Class{p}' for p in class_prefixes_sorted],
            yticklabels=[f'Class{p}' for p in class_prefixes_sorted],
            cbar_kws={'label': 'Prediction Ratio'}, annot_kws={'fontsize': 3})
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix - {MODE.upper()} Mode (unsaturated test)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_ROOT, 'test', 'confusion_matrix.png'), dpi=300)
plt.close()

# Save confusion matrix data to Excel
conf_df = pd.DataFrame(conf_matrix,
                       index=[f'True_{p}' for p in class_prefixes_sorted],
                       columns=[f'Pred_{p}' for p in class_prefixes_sorted])
conf_norm_df = pd.DataFrame(conf_matrix_norm,
                            index=[f'True_{p}' for p in class_prefixes_sorted],
                            columns=[f'Pred_{p}' for p in class_prefixes_sorted])
with pd.ExcelWriter(os.path.join(RESULT_ROOT, 'test', 'confusion_matrix.xlsx')) as writer:
    conf_df.to_excel(writer, sheet_name='Counts')
    conf_norm_df.to_excel(writer, sheet_name='Normalized')
print(f"Confusion matrix data saved to: {os.path.join(RESULT_ROOT, 'test', 'confusion_matrix.xlsx')}")

# Training curves
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title(f'Loss Curves - {MODE.upper()} Mode')
plt.savefig(os.path.join(RESULT_ROOT, 'all', 'loss_curves.png'))
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
plt.title(f'Accuracy Curves - {MODE.upper()} Mode')
plt.savefig(os.path.join(RESULT_ROOT, 'all', 'accuracy_curves.png'))
plt.close()

print(f"\nAll results saved to: {os.path.abspath(RESULT_ROOT)}")