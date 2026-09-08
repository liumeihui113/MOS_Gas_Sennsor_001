# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, average_precision_score)
import torch.nn.functional as F

# -------------------------- Global Parameters --------------------------
DATA_ROOT = r'Single-gas_dataset_4 batches'          # Root directory containing subfolders 1,2,3,4
RESULT_ROOT = 'Single-gas_classification_results_4CV'     # Root directory for saving results

SELECT_CHANNEL = [0, 1, 2, 3, 4, 5, 6, 7]
BATCH_SIZE = 64
EPOCHS = 10000
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.3
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 1
SEED_VALUE = 41
PLOT_DPI = 300

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # Keep for any possible Chinese in system, but labels are English
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed(SEED_VALUE)
torch.cuda.manual_seed_all(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------- 1. Data Loading --------------------------
print("=" * 60)
print("Starting data loading...")
print("=" * 60)

def extract_class_prefix(file_name):
    match = re.match(r'^(\d+)-', file_name)
    if not match:
        raise ValueError(f"File name {file_name} does not follow pattern 'number-'")
    return match.group(1)

all_files = []
for fold_name in ['1', '2', '3', '4']:
    fold_path = os.path.join(DATA_ROOT, fold_name)
    if not os.path.isdir(fold_path):
        continue
    for fname in os.listdir(fold_path):
        if fname.endswith('.xlsx'):
            file_path = os.path.join(fold_path, fname)
            try:
                cls = extract_class_prefix(fname)
                all_files.append((int(fold_name), file_path, cls))
            except ValueError as e:
                print(f"Skipping file {fname}: {e}")

class_file_dict = {}
for fold_id, fpath, cls in all_files:
    class_file_dict.setdefault(cls, []).append((fold_id, fpath))

sorted_classes = sorted(class_file_dict.keys(), key=int)
if sorted_classes != [str(i) for i in range(1, 11)]:
    raise ValueError(f"Incomplete classes, found: {sorted_classes}, need 1-10")

print("\nSample distribution per class across folds:")
for cls in sorted_classes:
    files = class_file_dict[cls]
    fold_counts = {}
    for fold_id, _ in files:
        fold_counts[fold_id] = fold_counts.get(fold_id, 0) + 1
    print(f"Class {cls}: {fold_counts}")

data_list, label_list, fold_list = [], [], []
class_label_map = {cls: idx for idx, cls in enumerate(sorted_classes)}
all_files_sorted = sorted(all_files, key=lambda x: (int(x[2]), x[0]))

for fold_id, fpath, cls in all_files_sorted:
    df = pd.read_excel(fpath).iloc[:, 1:]
    data_list.append(df.values)
    label_list.append(class_label_map[cls])
    fold_list.append(fold_id)

min_rows = min(sample.shape[0] for sample in data_list)
data_list = [sample[:min_rows] for sample in data_list]
data = np.array(data_list)
labels = np.array(label_list)
folds = np.array(fold_list)

n_samples = data.shape[0]
n_channels = len(SELECT_CHANNEL)
n_classes = 10

print(f"\nTotal samples: {n_samples}")
print(f"Each sample shape: {data[0].shape}")
print(f"Label distribution: {np.bincount(labels)}")
print(f"Samples per fold: {[np.sum(folds==i) for i in range(1,5)]}")

# -------------------------- 2. Data Standardization --------------------------
scaler = StandardScaler()
data_shape = data.shape
data_scaled = scaler.fit_transform(
    data.reshape(-1, data_shape[1] * data_shape[2])
).reshape(data_shape)

data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
full_dataset = TensorDataset(data_tensor, labels_tensor)

# -------------------------- 3. Model Definition --------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, in_channel=len(SELECT_CHANNEL), out_channel=10,
                 lstm_hidden_dim=LSTM_HIDDEN_DIM, lstm_layers=LSTM_LAYERS):
        super(CNN_LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers, batch_first=True)
        with torch.no_grad():
            dummy = torch.randn(1, data_tensor.shape[1], in_channel)
            dummy = dummy.permute(0,2,1)
            out = self.layer3(self.layer2(self.layer1(dummy)))
            out = out.permute(0,2,1)
            lstm_out, _ = self.lstm(out)
            self.fc_input_dim = lstm_out.reshape(1, -1).shape[1]
        self.fc1 = nn.Linear(self.fc_input_dim, 100)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(100, out_channel)

    def forward(self, x, return_features=False):
        x = x.permute(0,2,1)[:, SELECT_CHANNEL, :]
        x = self.layer1(x)
        x = self.layer2(x)
        cnn_features = self.layer3(x)
        x = cnn_features.permute(0,2,1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(lstm_out.size(0), -1)
        x = self.fc1(lstm_out)
        x = self.dropout(x)
        x = self.fc2(x)
        if return_features:
            return x, cnn_features
        return x

# -------------------------- 4. 4-Fold Cross Validation --------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

fold_results = []
all_test_targets = []   # collect true labels from all test folds
all_test_preds = []     # collect predicted labels from all test folds
all_test_probs = []     # collect prediction probabilities from all test folds

for test_fold in [1, 2, 3, 4]:
    print("\n" + "=" * 60)
    print(f"Starting fold {test_fold}: Test set = Folder {test_fold}")
    print("=" * 60)

    test_idx = np.where(folds == test_fold)[0]
    train_idx = np.where(folds != test_fold)[0]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    fold_result_dir = os.path.join(RESULT_ROOT, f'fold{test_fold}')
    os.makedirs(fold_result_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_result_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(fold_result_dir, 'all'), exist_ok=True)

    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_test_acc = 0.0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        test_loss = 0.0
        correct_test, total_test = 0, 0
        all_targets, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (pred == targets).sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        test_loss = test_loss / total_test
        test_acc = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(fold_result_dir, 'best_model_params.pth'))

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Load best model, re-evaluate and save results
    model.load_state_dict(torch.load(os.path.join(fold_result_dir, 'best_model_params.pth')))
    model.eval()
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Collect into overall lists
    all_test_targets.append(all_targets)
    all_test_preds.append(all_preds)
    all_test_probs.append(all_probs)

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    fold_results.append({'fold': test_fold, 'acc': acc, 'recall': recall, 'f1': f1})

    # Save metrics for this fold
    with open(os.path.join(fold_result_dir, 'test', 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Fold {test_fold} Test Metrics\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n")

    # Confusion matrix
    label_class_map = {idx: str(cls) for cls, idx in class_label_map.items()}
    class_names = [label_class_map[i] for i in range(10)]
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,8))
    plt.imshow(cm_norm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Fold {test_fold}')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                     color="white" if cm_norm[i,j] > 0.5 else "black")
    plt.ylabel('True Class'); plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_result_dir, 'test', 'confusion_matrix.png'), dpi=PLOT_DPI)
    plt.close()

    # Export confusion matrix data to Excel
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_excel(os.path.join(fold_result_dir, 'test', 'confusion_matrix.xlsx'))

    # PR curves
    y_bin = label_binarize(all_targets, classes=np.arange(10))
    plt.figure(figsize=(12,10))
    colors = plt.cm.get_cmap('tab10', 10)
    for i in range(10):
        precision, recall_curve, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_bin[:, i], all_probs[:, i])
        plt.plot(recall_curve, precision, color=colors(i), lw=2, label=f'Class {class_names[i]} (AP={ap:.3f})')
    p_micro, r_micro, _ = precision_recall_curve(y_bin.ravel(), all_probs.ravel())
    ap_micro = average_precision_score(y_bin, all_probs, average='micro')
    plt.plot(r_micro, p_micro, color='black', linestyle=':', linewidth=4, label=f'Micro-average (AP={ap_micro:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'PR Curve - Fold {test_fold}')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_result_dir, 'test', 'pr_curve.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # Training curves
    plt.figure(figsize=(12,5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Loss Curves - Fold {test_fold}')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fold_result_dir, 'all', 'loss_curves.png'), dpi=PLOT_DPI)
    plt.close()

    plt.figure(figsize=(12,5))
    plt.plot(range(1, EPOCHS+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, EPOCHS+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curves - Fold {test_fold}')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fold_result_dir, 'all', 'accuracy_curves.png'), dpi=PLOT_DPI)
    plt.close()

    # Predictions details for this fold
    pred_df = pd.DataFrame({
        'True Label': all_targets,
        'Predicted Label': all_preds,
        'True Class': [label_class_map[l] for l in all_targets],
        'Predicted Class': [label_class_map[l] for l in all_preds]
    })
    for i in range(10):
        pred_df[f'Class_{i}_Prob'] = all_probs[:, i]
    pred_df.to_csv(os.path.join(fold_result_dir, 'test', 'predictions.csv'), index=False, encoding='utf-8-sig')

# -------------------------- 5. Overall Results Summary --------------------------
print("\n" + "=" * 60)
print("4-Fold Cross-Validation Summary")
print("=" * 60)
for res in fold_results:
    print(f"Fold {res['fold']}: Acc={res['acc']:.4f}, Recall={res['recall']:.4f}, F1={res['f1']:.4f}")

acc_list = [r['acc'] for r in fold_results]
recall_list = [r['recall'] for r in fold_results]
f1_list = [r['f1'] for r in fold_results]
avg_acc = np.mean(acc_list)
std_acc = np.std(acc_list)
avg_recall = np.mean(recall_list)
std_recall = np.std(recall_list)
avg_f1 = np.mean(f1_list)
std_f1 = np.std(f1_list)

print(f"\nAverage performance: Acc={avg_acc:.4f}±{std_acc:.4f}, Recall={avg_recall:.4f}±{std_recall:.4f}, F1={avg_f1:.4f}±{std_f1:.4f}")

# Save per-fold metrics CSV
fold_metrics_df = pd.DataFrame(fold_results)
fold_metrics_df.to_csv(os.path.join(RESULT_ROOT, 'fold_metrics.csv'), index=False, encoding='utf-8-sig')

# Save summary text (with mean ± std)
with open(os.path.join(RESULT_ROOT, 'summary_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("4-Fold Cross-Validation Summary\n")
    f.write("="*40 + "\n")
    for res in fold_results:
        f.write(f"Fold {res['fold']}: Acc={res['acc']:.4f}, Recall={res['recall']:.4f}, F1={res['f1']:.4f}\n")
    f.write(f"\nMean ± Std:\n")
    f.write(f"Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n")
    f.write(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}\n")
    f.write(f"F1-Score: {avg_f1:.4f} ± {std_f1:.4f}\n")

# ---- Overall Confusion Matrix ----
# Merge all test predictions
all_targets_total = np.concatenate(all_test_targets)
all_preds_total = np.concatenate(all_test_preds)
all_probs_total = np.concatenate(all_test_probs, axis=0)

# Overall confusion matrix
cm_total = confusion_matrix(all_targets_total, all_preds_total)
cm_total_norm = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10,8))
plt.imshow(cm_total_norm, cmap=plt.cm.Blues)
plt.title('Overall Confusion Matrix (All Test Folds Combined)')
plt.colorbar()
tick_marks = np.arange(10)
class_names = [label_class_map[i] for i in range(10)]
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
for i in range(10):
    for j in range(10):
        plt.text(j, i, f"{cm_total[i,j]}", ha="center", va="center",
                 color="white" if cm_total_norm[i,j] > 0.5 else "black")
plt.ylabel('True Class'); plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_ROOT, 'overall_confusion_matrix.png'), dpi=PLOT_DPI)
plt.close()

# Also save confusion matrix data as CSV
cm_df = pd.DataFrame(cm_total, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(RESULT_ROOT, 'overall_confusion_matrix.csv'), encoding='utf-8-sig')

# ---- Overall PR Curve ----
y_bin_total = label_binarize(all_targets_total, classes=np.arange(10))
plt.figure(figsize=(12,10))
colors = plt.cm.get_cmap('tab10', 10)
for i in range(10):
    precision, recall_curve, _ = precision_recall_curve(y_bin_total[:, i], all_probs_total[:, i])
    ap = average_precision_score(y_bin_total[:, i], all_probs_total[:, i])
    plt.plot(recall_curve, precision, color=colors(i), lw=2, label=f'Class {class_names[i]} (AP={ap:.3f})')
p_micro, r_micro, _ = precision_recall_curve(y_bin_total.ravel(), all_probs_total.ravel())
ap_micro = average_precision_score(y_bin_total, all_probs_total, average='micro')
plt.plot(r_micro, p_micro, color='black', linestyle=':', linewidth=4, label=f'Micro-average (AP={ap_micro:.3f})')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Overall PR Curve (All Test Folds Combined)')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_ROOT, 'overall_pr_curve.png'), dpi=PLOT_DPI, bbox_inches='tight')
plt.close()

# ---- Overall Predictions CSV ----
total_pred_df = pd.DataFrame({
    'True Label': all_targets_total,
    'Predicted Label': all_preds_total,
    'True Class': [label_class_map[l] for l in all_targets_total],
    'Predicted Class': [label_class_map[l] for l in all_preds_total]
})
for i in range(10):
    total_pred_df[f'Class_{i}_Prob'] = all_probs_total[:, i]
total_pred_df.to_csv(os.path.join(RESULT_ROOT, 'overall_predictions.csv'), index=False, encoding='utf-8-sig')

# -------------------------- 6. Completion Message --------------------------
print("\n" + "=" * 80)
print("All tasks completed!")
print(f"Results saved in: {os.path.abspath(RESULT_ROOT)}")
print("=" * 80)
print("Key result files:")
print(" - fold_*/ : Detailed results per fold (model, metrics, curves, etc.)")
print(" - summary_metrics.txt : Average metrics ± std across 4 folds")
print(" - overall_confusion_matrix.* : Overall confusion matrix (heatmap + Excel)")
print(" - overall_pr_curve.png : Overall PR curve")
print(" - overall_predictions.csv : Prediction details for all test samples")
print(" - fold_metrics.csv : List of metrics per fold")
print("=" * 80)