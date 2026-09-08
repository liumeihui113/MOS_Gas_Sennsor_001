import pandas as pd
import os
import re
import numpy as np
import scipy.fft as fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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

# -------------------------- 1. Data Loading (group by prefix, 10 samples per class) --------------------------
folder_path = r'Odour_saturated_dataset'  # Contains 103 classes, we take first 100

file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
if not file_list:
    raise FileNotFoundError(f"No Excel files found in {folder_path}")

def extract_class_prefix(file_name):
    match = re.match(r'^(\d+)_', file_name)
    if not match:
        raise ValueError(f"File name {file_name} does not start with 'number_'")
    return match.group(1)

# Group by prefix
class_file_dict = {}
for file_name in file_list:
    prefix = extract_class_prefix(file_name)
    class_file_dict.setdefault(prefix, []).append(file_name)

# Filter only classes 1-100 (ignore 101, 102, 103)
filtered_prefixes = [p for p in class_file_dict.keys() if int(p) <= 100]
sorted_class_prefixes = sorted(filtered_prefixes, key=int)

# Validate
n_classes = len(sorted_class_prefixes)
samples_per_class = 10
if n_classes != 100:
    raise ValueError(f"Expected 100 classes, got {n_classes}")
for prefix in sorted_class_prefixes:
    if len(class_file_dict[prefix]) != samples_per_class:
        raise ValueError(f"Class {prefix} has {len(class_file_dict[prefix])} files, expected 10")

# Label mapping: class -> 0~99
class_label_map = {prefix: idx for idx, prefix in enumerate(sorted_class_prefixes)}
label_class_map = {idx: prefix for prefix, idx in class_label_map.items()}

# Load data (keep raw, no standardization yet)
data = []   # each sample: [time steps, 24]
labels = []
for prefix in sorted_class_prefixes:
    class_files = class_file_dict[prefix]
    # Sort by file index (e.g., 1_1.xlsx, 1_2.xlsx, ...)
    def extract_file_index(fname):
        match = re.search(r'_(\d+)\.xlsx$', fname)   # new naming without "拐点"
        return int(match.group(1)) if match else 0
    class_files_sorted = sorted(class_files, key=extract_file_index)
    for fname in class_files_sorted:
        file_path = os.path.join(folder_path, fname)
        df = pd.read_excel(file_path).iloc[:, 1:]   # skip first column (index)
        data.append(df.values)
        labels.append(class_label_map[prefix])

# Unify length (crop to shortest)
min_rows = min(len(s) for s in data)
data = np.array([s[:min_rows] for s in data])   # shape: [1000, T, 24]
labels = np.array(labels)                       # shape: [1000]

print("=" * 60)
print("Data loading completed")
print(f"Total samples: {len(data)} (100 classes, 10 each)")
print(f"Sample shape: {data.shape[1:]} (time steps × channels)")
print("=" * 60)

# -------------------------- 2. Model Definition (CNN-LSTM) --------------------------
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
        # Dynamically compute FC input dimension
        with torch.no_grad():
            test_input = torch.randn(1, data.shape[1], in_channel)
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
        # x: [batch, T, 24]
        x = x.permute(0, 2, 1)  # [batch, 24, T]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1)  # [batch, T', 64]
        lstm_out, _ = self.lstm(x)  # [batch, T', 128]
        lstm_out = lstm_out.contiguous().view(lstm_out.size(0), -1)
        x = self.fc1(lstm_out)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------- 3. Training and Evaluation Function (per fold) --------------------------
def train_and_evaluate(fold_idx, train_indices, test_indices, data_tensor, labels_tensor,
                       result_root, device, epochs=300, batch_size=64):
    """
    Train and evaluate on one fold, save all results for that fold.
    Returns test targets, predictions, and probabilities.
    """
    fold_dir = os.path.join(result_root, f'fold_{fold_idx+1}')
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'all'), exist_ok=True)

    # ---- Data preparation ----
    train_data = data_tensor[train_indices]  # [n_train, T, 24]
    test_data = data_tensor[test_indices]    # [n_test, T, 24]
    train_labels = labels_tensor[train_indices]
    test_labels = labels_tensor[test_indices]

    # Standardize based on training set mean/std
    train_flat = train_data.view(train_data.size(0), -1)
    mean = train_flat.mean(dim=0, keepdim=True)
    std = train_flat.std(dim=0, unbiased=False, keepdim=True)
    std[std < 1e-8] = 1.0
    train_flat_norm = (train_flat - mean) / std
    train_data_norm = train_flat_norm.view(train_data.shape)

    test_flat = test_data.view(test_data.size(0), -1)
    test_flat_norm = (test_flat - mean) / std
    test_data_norm = test_flat_norm.view(test_data.shape)

    train_dataset = TensorDataset(train_data_norm, train_labels)
    test_dataset = TensorDataset(test_data_norm, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---- Model initialization ----
    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # ---- Training loop ----
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (pred == targets).sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        test_loss = running_loss / total
        test_acc = correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model_params.pth'))

        if (epoch+1) % 10 == 0:
            print(f"Fold {fold_idx+1} Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # ---- Load best model for final evaluation ----
    model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model_params.pth')))
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
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

    # ---- Compute metrics and save ----
    acc = accuracy_score(all_targets, all_preds)
    rec = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    metrics_str = (f"Fold {fold_idx+1} Test Metrics\n"
                   f"Accuracy: {acc:.4f}\n"
                   f"Recall (weighted): {rec:.4f}\n"
                   f"F1-Score (weighted): {f1:.4f}\n")
    print(metrics_str)
    with open(os.path.join(fold_dir, 'test', 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(metrics_str)

    # Confusion matrix
    class_prefixes = [label_class_map[i] for i in range(100)]
    cm = confusion_matrix(all_targets, all_preds, labels=range(100))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Save Excel
    cm_raw_df = pd.DataFrame(cm, index=[f'True_{p}' for p in class_prefixes],
                             columns=[f'Pred_{p}' for p in class_prefixes])
    cm_norm_df = pd.DataFrame(np.round(cm_norm, 4), index=[f'True_{p}' for p in class_prefixes],
                              columns=[f'Pred_{p}' for p in class_prefixes])
    with pd.ExcelWriter(os.path.join(fold_dir, 'test', 'confusion_matrix.xlsx'), engine='openpyxl') as writer:
        cm_raw_df.to_excel(writer, sheet_name='Raw Counts')
        cm_norm_df.to_excel(writer, sheet_name='Normalized')
    # Heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=[f'Class{p}' for p in class_prefixes],
                yticklabels=[f'Class{p}' for p in class_prefixes],
                cbar_kws={'label': 'Prediction Ratio'}, annot_kws={'fontsize': 3})
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Fold {fold_idx+1} Test Confusion Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'test', 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Training curves
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_idx+1} Loss Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fold_dir, 'all', 'loss_curves.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs+1), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Fold {fold_idx+1} Accuracy Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fold_dir, 'all', 'accuracy_curves.png'), dpi=300)
    plt.close()

    # Save training process data
    train_df = pd.DataFrame({
        'Epoch': range(1, epochs+1),
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Train Accuracy': train_accs,
        'Test Accuracy': test_accs
    })
    train_df.to_csv(os.path.join(fold_dir, 'all', 'train_process_data.csv'), index=False, encoding='utf-8-sig')

    # PR curves (selected key classes + Micro-average)
    y_true_bin = label_binarize(all_targets, classes=range(100))
    precision = {}
    recall = {}
    ap = {}
    plt.figure(figsize=(14, 12))
    colors = plt.cm.get_cmap('tab20b', 100)
    # Select classes to display: first 10, last 10, top5 AP, bottom5 AP
    ap_list = [(i, average_precision_score(y_true_bin[:, i], all_probs[:, i])) for i in range(100)]
    ap_list_sorted = sorted(ap_list, key=lambda x: x[1], reverse=True)
    show_labels = list(range(10)) + list(range(90, 100)) + [x[0] for x in ap_list_sorted[:5]] + [x[0] for x in ap_list_sorted[-5:]]
    show_labels = sorted(set(show_labels))
    pr_data = {'Class Prefix': [], 'Recall': [], 'Precision': [], 'Average Precision (AP)': []}
    for label in range(100):
        p, r, _ = precision_recall_curve(y_true_bin[:, label], all_probs[:, label])
        precision[label], recall[label] = p, r
        ap[label] = average_precision_score(y_true_bin[:, label], all_probs[:, label])
        pr_data['Class Prefix'].extend([f'Class{label_class_map[label]}'] * len(r))
        pr_data['Recall'].extend(r)
        pr_data['Precision'].extend(p)
        pr_data['Average Precision (AP)'].extend([ap[label]] * len(r))
        if label in show_labels:
            plt.plot(r, p, color=colors(label), lw=2, alpha=0.8,
                     label=f'Class{label_class_map[label]} (AP={ap[label]:.3f})')
    # Micro-average
    p_micro, r_micro, _ = precision_recall_curve(y_true_bin.ravel(), all_probs.ravel())
    ap_micro = average_precision_score(y_true_bin, all_probs, average='micro')
    plt.plot(r_micro, p_micro, color='black', linestyle=':', linewidth=4,
             label=f'Micro-average (AP={ap_micro:.3f})')
    pr_data['Class Prefix'].extend(['Micro-average'] * len(r_micro))
    pr_data['Recall'].extend(r_micro)
    pr_data['Precision'].extend(p_micro)
    pr_data['Average Precision (AP)'].extend([ap_micro] * len(r_micro))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'Fold {fold_idx+1} PR Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'test', 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Save PR data
    pd.DataFrame(pr_data).to_csv(os.path.join(fold_dir, 'test', 'pr_curve_data.csv'), index=False, encoding='utf-8-sig')

    # Note: predictions_vs_targets.csv has been removed as requested.

    return all_targets, all_preds, all_probs


# -------------------------- 4. Main: 5-fold Cross Validation --------------------------
if __name__ == "__main__":
    # Set random seeds
    seed_value = 41
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert data to tensors (raw, unstandardized)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create result root directory
    RESULT_ROOT = 'Odour_saturated_results_100_5CV'
    os.makedirs(RESULT_ROOT, exist_ok=True)

    # 5-fold stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []  # store (targets, preds, probs) for each fold

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data_tensor, labels_tensor)):
        print(f"\n{'='*60}\nStarting Fold {fold_idx+1}\n{'='*60}")
        fold_targets, fold_preds, fold_probs = train_and_evaluate(
            fold_idx, train_idx, test_idx, data_tensor, labels_tensor,
            RESULT_ROOT, device, epochs=300, batch_size=64
        )
        fold_results.append((fold_targets, fold_preds, fold_probs))

    # -------------------------- 5. Summarize all folds --------------------------
    print("\n" + "="*60)
    print("Summary of 5-fold Cross-Validation")
    print("="*60)

    # Per-fold metrics
    metrics = []
    for i, (targets, preds, _) in enumerate(fold_results):
        acc = accuracy_score(targets, preds)
        rec = recall_score(targets, preds, average='weighted')
        f1 = f1_score(targets, preds, average='weighted')
        metrics.append({'Fold': i+1, 'Accuracy': acc, 'Recall': rec, 'F1': f1})
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv(os.path.join(RESULT_ROOT, 'fold_metrics.csv'), index=False, encoding='utf-8-sig')

    # Mean ± std
    mean_acc = metrics_df['Accuracy'].mean()
    std_acc = metrics_df['Accuracy'].std()
    mean_rec = metrics_df['Recall'].mean()
    std_rec = metrics_df['Recall'].std()
    mean_f1 = metrics_df['F1'].mean()
    std_f1 = metrics_df['F1'].std()
    summary_str = (f"5-fold CV summary (mean ± std)\n"
                   f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n"
                   f"Recall: {mean_rec:.4f} ± {std_rec:.4f}\n"
                   f"F1-Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
    print(summary_str)
    with open(os.path.join(RESULT_ROOT, 'summary_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_str)

    # Combine all test predictions (each sample appears exactly once)
    all_targets_total = np.concatenate([r[0] for r in fold_results])
    all_preds_total = np.concatenate([r[1] for r in fold_results])
    all_probs_total = np.concatenate([r[2] for r in fold_results])

    # Overall confusion matrix
    cm_total = confusion_matrix(all_targets_total, all_preds_total, labels=range(100))
    cm_norm_total = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis]
    class_prefixes = [label_class_map[i] for i in range(100)]
    # Save Excel
    cm_raw_df = pd.DataFrame(cm_total, index=[f'True_{p}' for p in class_prefixes],
                             columns=[f'Pred_{p}' for p in class_prefixes])
    cm_norm_df = pd.DataFrame(np.round(cm_norm_total, 4), index=[f'True_{p}' for p in class_prefixes],
                              columns=[f'Pred_{p}' for p in class_prefixes])
    with pd.ExcelWriter(os.path.join(RESULT_ROOT, 'overall_confusion_matrix.xlsx'), engine='openpyxl') as writer:
        cm_raw_df.to_excel(writer, sheet_name='Raw Counts')
        cm_norm_df.to_excel(writer, sheet_name='Normalized')
    # Heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_norm_total, annot=cm_total, fmt='d', cmap='Blues',
                xticklabels=[f'Class{p}' for p in class_prefixes],
                yticklabels=[f'Class{p}' for p in class_prefixes],
                cbar_kws={'label': 'Prediction Ratio'}, annot_kws={'fontsize': 3})
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Overall Confusion Matrix (5-fold CV)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_ROOT, 'overall_confusion_matrix.png'), dpi=300)
    plt.close()

    # Overall PR curves
    y_true_bin_total = label_binarize(all_targets_total, classes=range(100))
    plt.figure(figsize=(14, 12))
    colors = plt.cm.get_cmap('tab20b', 100)
    ap_list_total = [(i, average_precision_score(y_true_bin_total[:, i], all_probs_total[:, i])) for i in range(100)]
    ap_sorted = sorted(ap_list_total, key=lambda x: x[1], reverse=True)
    show_labels = list(range(10)) + list(range(90, 100)) + [x[0] for x in ap_sorted[:5]] + [x[0] for x in ap_sorted[-5:]]
    show_labels = sorted(set(show_labels))
    for label in range(100):
        p, r, _ = precision_recall_curve(y_true_bin_total[:, label], all_probs_total[:, label])
        ap = average_precision_score(y_true_bin_total[:, label], all_probs_total[:, label])
        if label in show_labels:
            plt.plot(r, p, color=colors(label), lw=2, alpha=0.8,
                     label=f'Class{label_class_map[label]} (AP={ap:.3f})')
    # Micro
    p_micro, r_micro, _ = precision_recall_curve(y_true_bin_total.ravel(), all_probs_total.ravel())
    ap_micro = average_precision_score(y_true_bin_total, all_probs_total, average='micro')
    plt.plot(r_micro, p_micro, color='black', linestyle=':', linewidth=4,
             label=f'Micro-average (AP={ap_micro:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Overall PR Curves (5-fold CV)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_ROOT, 'overall_pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save overall predictions (no per-fold predictions CSV)
    overall_pred_df = pd.DataFrame({
        'Global Sample Index': np.arange(len(all_targets_total)),
        'True Label': all_targets_total,
        'Predicted Label': all_preds_total,
        'True Class Prefix': [label_class_map[l] for l in all_targets_total],
        'Predicted Class Prefix': [label_class_map[l] for l in all_preds_total]
    })
    for label in range(100):
        overall_pred_df[f'Class{label_class_map[label]}_Prob'] = all_probs_total[:, label]
    overall_pred_df.to_csv(os.path.join(RESULT_ROOT, 'overall_predictions.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "="*80)
    print("All tasks completed! Results saved in:", os.path.abspath(RESULT_ROOT))
    print("="*80)
    print("Key result files:")
    print(" - fold_*/ : detailed results per fold (model, metrics, curves, etc.)")
    print(" - summary_metrics.txt : average metrics ± std across 5 folds")
    print(" - overall_confusion_matrix.* : overall confusion matrix (heatmap + Excel)")
    print(" - overall_pr_curve.png : overall PR curve")
    print(" - overall_predictions.csv : prediction details for all test samples")
    print(" - fold_metrics.csv : metrics per fold")
    print("="*80)