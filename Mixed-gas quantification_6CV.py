# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# -------------------------- Global Parameters --------------------------
folder_path = r'Mixed-gas_dataset'                    # data folder
concentration_file = 'Mixing Ratio.xlsx'             # concentration labels file
total_classes = 100                                  # total number of classes
select_channel = [0, 1, 2, 3, 4, 5, 6, 7]            # all 8 channels
epochs = 1500
batch_size = 64
learning_rate = 3e-4
weight_decay = 1e-4
latent_dim = 256
augment_factor = 3                                   # augmentation factor per original sample

# Augmentation parameters
gaussian_noise_std = 0.1
time_stretch_range = (0.9, 1.1)
smooth_sigma = 1.0

k_folds = 6                                          # leave-one-out per class
selected_classes_num = 20                            # number of classes to display

RESULT_ROOT = 'Mixed-gas_quantification_results_6CV'     # root directory for all results

# -------------------------- Utility Functions --------------------------
def extract_class_prefix(file_name):
    match = re.match(r'^(\d+)_', file_name)
    if not match:
        raise ValueError(f"File name {file_name} does not match pattern")
    return match.group(1)

def extract_file_index(file_name):
    # New naming: e.g., "1_1.xlsx" -> extract trailing number
    match = re.search(r'_(\d+)\.xlsx$', file_name)
    return int(match.group(1)) if match else 0

def extract_time_domain_features_torch(seq):
    """Extract time-domain statistical features using PyTorch (differentiable)"""
    features = []
    for channel in range(seq.shape[1]):
        data = seq[:, channel]
        features.append(torch.mean(data))
        features.append(torch.std(data))
        features.append(torch.max(data) - torch.min(data))
        features.append(torch.sqrt(torch.mean(data ** 2)))
        features.append(torch.sum(torch.abs(torch.diff(data))))
        features.append(torch.argmax(data).float() / len(data))
    return torch.stack(features)

# -------------------------- Data Augmentation Dataset --------------------------
class RawAugmentedDataset(Dataset):
    """
    Generate augmented samples from raw data.
    Methods:
      1. Gaussian noise: x' = x + ε, ε ~ N(0, σ²)
      2. Time stretching: linear interpolation with factor s ∈ [0.9,1.1], then crop/pad
      3. Gaussian smoothing: 1D Gaussian filter with sigma = smooth_sigma
    One method is randomly selected per sample.
    """
    def __init__(self, data, targets, augment=True):
        self.data = data          # (N, T, C)
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.data) * (augment_factor if self.augment else 1)

    def __getitem__(self, idx):
        base_idx = idx // (augment_factor if self.augment else 1)
        data = self.data[base_idx].copy()
        target = self.targets[base_idx].copy()

        if self.augment:
            aug_type = np.random.choice(['noise', 'stretch', 'smooth'])

            if aug_type == 'noise':
                noise = np.random.normal(0, gaussian_noise_std, data.shape)
                data = data + noise

            elif aug_type == 'stretch':
                T = data.shape[0]
                stretch_factor = np.random.uniform(*time_stretch_range)
                new_T = int(T * stretch_factor)
                stretched = np.zeros((new_T, data.shape[1]))
                for c in range(data.shape[1]):
                    stretched[:, c] = np.interp(
                        np.linspace(0, T - 1, new_T),
                        np.arange(T),
                        data[:, c]
                    )
                if new_T > T:
                    data = stretched[:T]
                else:
                    data = np.pad(stretched, ((0, T - new_T), (0, 0)), mode='edge')

            elif aug_type == 'smooth':
                for c in range(data.shape[1]):
                    data[:, c] = gaussian_filter1d(data[:, c], sigma=smooth_sigma)

        return data, target

# -------------------------- CNN-LSTM Model --------------------------
class CNNLSTMRegressor(nn.Module):
    def __init__(self, in_channel, seq_len, output_dim, latent_dim=256):
        super(CNNLSTMRegressor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
        )
        with torch.no_grad():
            test_input = torch.randn(1, in_channel, seq_len)
            cnn_out = self.cnn(test_input)
            self.lstm_input_dim = cnn_out.size(1)
            self.lstm_seq_len = cnn_out.size(2)

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * 2, 1),
            nn.Softmax(dim=1)
        )
        self.statistic_fc = nn.Sequential(
            nn.Linear(len(select_channel) * 6, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim * 2 + 128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)[:, select_channel, :]
        cnn_feat = self.cnn(x_cnn)
        cnn_feat = cnn_feat.permute(0, 2, 1)
        lstm_feat, _ = self.lstm(cnn_feat)
        attn_weights = self.attention(lstm_feat)
        lstm_attn = torch.sum(lstm_feat * attn_weights, dim=1)

        batch_size = x.size(0)
        stat_features = []
        for i in range(batch_size):
            stat_feat = extract_time_domain_features_torch(x[i])
            stat_features.append(stat_feat)
        stat_features = torch.stack(stat_features)
        stat_feat = self.statistic_fc(stat_features)

        fused = torch.cat([lstm_attn, stat_feat], dim=1)
        return self.regressor(fused)

# -------------------------- Training Function --------------------------
def train_model(model, train_loader, device, epochs, criterion, optimizer, scheduler, fold_dir):
    best_loss = float('inf')
    patience = 50
    counter = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        lr = optimizer.param_groups[0]['lr']
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))
            counter = 0
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.6f} | *Best* | LR: {lr:.8f}")
        else:
            counter += 1
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.6f} | LR: {lr:.8f}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save training loss data
    pd.DataFrame({
        'Epoch': range(1, len(train_losses)+1),
        'TrainLoss': train_losses
    }).to_csv(os.path.join(fold_dir, 'train_loss.csv'), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fold_dir, 'train_loss.png'), dpi=200)
    plt.close()

# -------------------------- Evaluation Function --------------------------
def evaluate_fold(model, test_loader, device, concentration_columns, scaler_y, fold_dir, test_labels):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Inverse transform
    pred_orig = scaler_y.inverse_transform(all_preds)
    true_orig = scaler_y.inverse_transform(all_targets)
    residuals = pred_orig - true_orig

    # Overall metrics
    mse = mean_squared_error(true_orig, pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_orig, pred_orig)
    r2 = r2_score(true_orig, pred_orig)

    # Per-component metrics
    comp_metrics = []
    for i, comp in enumerate(concentration_columns):
        comp_mse = mean_squared_error(true_orig[:, i], pred_orig[:, i])
        comp_rmse = np.sqrt(comp_mse)
        comp_mae = mean_absolute_error(true_orig[:, i], pred_orig[:, i])
        comp_r2 = r2_score(true_orig[:, i], pred_orig[:, i])
        comp_metrics.append({'Component': comp, 'RMSE': comp_rmse, 'MAE': comp_mae, 'R2': comp_r2})

    # Save predictions
    pred_df = pd.DataFrame({'Class': test_labels})
    for i, comp in enumerate(concentration_columns):
        pred_df[f'{comp}_Actual'] = true_orig[:, i]
        pred_df[f'{comp}_Predicted'] = pred_orig[:, i]
        pred_df[f'{comp}_Residual'] = residuals[:, i]
    pred_df.to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')

    # Residual distribution plot
    plt.figure(figsize=(8,5))
    sns.histplot(residuals.flatten(), kde=True, bins=30)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residual (ppb)')
    plt.title(f'Residual Distribution (RMSE={rmse:.3f})')
    plt.savefig(os.path.join(fold_dir, 'residuals.png'), dpi=200)
    plt.close()

    # Scatter plots per component
    n_comp = len(concentration_columns)
    n_rows = (n_comp + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    for i, comp in enumerate(concentration_columns):
        ax = axes[i]
        ax.scatter(true_orig[:, i], pred_orig[:, i], alpha=0.6, s=20)
        minv = min(true_orig[:, i].min(), pred_orig[:, i].min())
        maxv = max(true_orig[:, i].max(), pred_orig[:, i].max())
        ax.plot([minv, maxv], [minv, maxv], 'r--')
        ax.set_xlabel(f'Actual {comp} (ppb)')
        ax.set_ylabel(f'Predicted {comp} (ppb)')
        ax.set_title(f'{comp}  R²={comp_metrics[i]["R2"]:.4f}')
        ax.grid(alpha=0.3)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'scatter.png'), dpi=200)
    plt.close()

    # Residual distribution data (Excel)
    residuals_list = []
    for idx, label in enumerate(test_labels):
        for j, comp in enumerate(concentration_columns):
            residuals_list.append({
                'Class': label,
                'Component': comp,
                'Residual': residuals[idx, j]
            })
    pd.DataFrame(residuals_list).to_excel(
        os.path.join(fold_dir, 'residual_distribution_data.xlsx'),
        index=False
    )

    # Selected classes summary and bar charts
    unique_classes = sorted(set(test_labels))
    np.random.seed(41)
    if selected_classes_num < len(unique_classes):
        selected_classes = np.random.choice(unique_classes, selected_classes_num, replace=False)
    else:
        selected_classes = unique_classes

    # Build summary data
    all_class_data = []
    for cls in selected_classes:
        idx = test_labels.index(cls)   # only one sample per class in test
        row = {'Class': cls}
        for j, comp in enumerate(concentration_columns):
            row[f'{comp}_Actual'] = true_orig[idx, j]
            row[f'{comp}_Predicted'] = pred_orig[idx, j]
            row[f'{comp}_Residual'] = residuals[idx, j]
        all_class_data.append(row)
    pd.DataFrame(all_class_data).to_excel(
        os.path.join(fold_dir, 'selected_classes_summary.xlsx'),
        index=False
    )

    # Bar charts for each selected class
    class_dir = os.path.join(fold_dir, 'class_bar_charts')
    os.makedirs(class_dir, exist_ok=True)
    for cls in selected_classes:
        idx = test_labels.index(cls)
        avg_targets = true_orig[idx, :]
        avg_preds = pred_orig[idx, :]
        x = np.arange(len(concentration_columns))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, avg_targets, width, label='Actual')
        plt.bar(x + width/2, avg_preds, width, label='Predicted')
        plt.xlabel('Gas Component')
        plt.ylabel('Concentration (ppb)')
        plt.title(f'Class {cls} - Concentration Prediction Comparison')
        plt.xticks(x, concentration_columns, rotation=45, ha='right')
        plt.legend()
        for i, (t, p) in enumerate(zip(avg_targets, avg_preds)):
            plt.text(i - width/2, t + 0.5, f'{t:.2f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + width/2, p + 0.5, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, f'class_{cls}_comparison.png'), dpi=300)
        plt.close()

    # Save component metrics
    pd.DataFrame(comp_metrics).to_csv(os.path.join(fold_dir, 'component_metrics.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(fold_dir, 'summary.txt'), 'w') as f:
        f.write(f"Overall: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}\n")
        for m in comp_metrics:
            f.write(f"{m['Component']}: RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}\n")

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'component_metrics': comp_metrics}

# -------------------------- Main Program --------------------------
def main():
    # Fix random seeds
    torch.manual_seed(41)
    np.random.seed(41)
    torch.backends.cudnn.deterministic = True

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print("="*70)
    print("6-fold Cross-Validation (leave-one-out per class, 6 samples each)")
    print("Training: 5 original + augmented samples (3x) per class")
    print("Test: original experimental data, no augmentation")
    print("Augmentation methods:")
    print(f"  1. Gaussian noise: ε~N(0, {gaussian_noise_std}²)")
    print(f"  2. Time stretching: factor ∈ [{time_stretch_range[0]}, {time_stretch_range[1]}]")
    print(f"  3. Gaussian smoothing: σ = {smooth_sigma}")
    print("="*70)

    # Load concentration file (located in folder_path)
    conc_file_path = os.path.join(folder_path, concentration_file)
    conc_df = pd.read_excel(conc_file_path)
    concentration_columns = conc_df.columns[1:].tolist()

    # Load sample files, EXCLUDING the concentration file
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and f != concentration_file]
    class_files_dict = {}
    for f in file_list:
        prefix = extract_class_prefix(f)
        class_files_dict.setdefault(prefix, []).append(f)

    sorted_prefixes = sorted(class_files_dict.keys(), key=int)
    if len(sorted_prefixes) != total_classes:
        raise ValueError(f"Expected {total_classes} classes, found {len(sorted_prefixes)}")

    # Build full dataset: 6 samples per class, sorted by index
    full_data, full_targets, full_labels = [], [], []
    class_index_range = {}

    idx = 0
    for prefix in sorted_prefixes:
        files = sorted(class_files_dict[prefix], key=extract_file_index)
        if len(files) != 6:
            raise ValueError(f"Class {prefix} must have 6 samples, found {len(files)}")
        conc_row = conc_df[conc_df['No.'] == int(prefix)]
        if conc_row.empty:
            raise ValueError(f"Concentration record for class {prefix} not found")
        conc_vals = conc_row.iloc[0, 1:].values

        start = idx
        for f in files:
            df = pd.read_excel(os.path.join(folder_path, f))
            data = df.iloc[:, 1:].values
            full_data.append(data)
            full_targets.append(conc_vals)
            full_labels.append(prefix)
            idx += 1
        class_index_range[prefix] = (start, idx)

    # Unify sequence length (crop)
    min_len = min(len(s) for s in full_data)
    full_data = np.array([s[:min_len] for s in full_data])   # (600, T, 8)
    full_targets = np.array(full_targets)                    # (600, 10)
    print(f"Total samples: {full_data.shape[0]}, time steps: {full_data.shape[1]}, channels: {full_data.shape[2]}")

    # Generate 6-fold indices: shuffle within each class, then take k-th as test (k=0..5)
    np.random.seed(41)
    fold_test_indices = []
    for fold in range(k_folds):
        fold_test = []
        for prefix in sorted_prefixes:
            start, end = class_index_range[prefix]
            indices = list(range(start, end))
            np.random.shuffle(indices)
            fold_test.append(indices[fold])
        fold_test_indices.append(fold_test)

    # Cross-validation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = os.path.join(RESULT_ROOT, f'CV6fold_{total_classes}classes_aug{augment_factor}x_{timestamp}')
    os.makedirs(result_root, exist_ok=True)

    all_fold_metrics = []

    for fold_idx, test_idx_list in enumerate(fold_test_indices):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx+1}/{k_folds}   Test samples: {len(test_idx_list)}")

        # Training indices (exclude test)
        train_idx_list = [i for i in range(full_data.shape[0]) if i not in test_idx_list]

        X_train_raw = full_data[train_idx_list]
        y_train_raw = full_targets[train_idx_list]
        X_test_raw = full_data[test_idx_list]
        y_test_raw = full_targets[test_idx_list]
        test_labels = [full_labels[i] for i in test_idx_list]

        # Augment training set
        print(f"  Generating augmented training set (original: {len(X_train_raw)}, augment factor: {augment_factor})")
        train_aug_data = list(X_train_raw)
        train_aug_targets = list(y_train_raw)
        aug_dataset = RawAugmentedDataset(X_train_raw, y_train_raw, augment=True)
        for i in range(len(aug_dataset)):
            data, target = aug_dataset[i]
            train_aug_data.append(data)
            train_aug_targets.append(target)
        X_train_aug = np.array(train_aug_data)
        y_train_aug = np.array(train_aug_targets)

        # Standardization (fit only on training set)
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_x.fit_transform(
            X_train_aug.reshape(-1, X_train_aug.shape[1]*X_train_aug.shape[2])
        ).reshape(X_train_aug.shape)
        X_test_scaled = scaler_x.transform(
            X_test_raw.reshape(-1, X_test_raw.shape[1]*X_test_raw.shape[2])
        ).reshape(X_test_raw.shape)

        y_train_scaled = scaler_y.fit_transform(y_train_aug)
        y_test_scaled = scaler_y.transform(y_test_raw)

        # DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test_scaled, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CNNLSTMRegressor(
            in_channel=len(select_channel),
            seq_len=full_data.shape[1],
            output_dim=full_targets.shape[1],
            latent_dim=latent_dim
        ).to(device)

        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-7)
        criterion = lambda pred, true: torch.where(
            torch.abs(pred - true) <= 1.0,
            0.5 * (pred - true)**2,
            1.0 * (torch.abs(pred - true) - 0.5)
        ).mean()   # Huber loss with delta=1.0

        fold_dir = os.path.join(result_root, f'fold_{fold_idx+1}')
        os.makedirs(fold_dir, exist_ok=True)

        # Train
        print("  Training...")
        train_model(model, train_loader, device, epochs, criterion, optimizer, scheduler, fold_dir)

        # Load best model and evaluate
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pth')))
        metrics = evaluate_fold(model, test_loader, device, concentration_columns,
                                scaler_y, fold_dir, test_labels)
        all_fold_metrics.append(metrics)
        print(f"  Fold {fold_idx+1} overall: RMSE={metrics['RMSE']:.6f}, R2={metrics['R2']:.6f}")

    # ---- Summarize all folds ----
    print("\n" + "="*70)
    print("6-fold Cross-Validation Summary")
    print("="*70)

    overall_keys = ['MSE', 'RMSE', 'MAE', 'R2']
    summary_stats = {}
    for key in overall_keys:
        vals = [m[key] for m in all_fold_metrics]
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        summary_stats[key] = (mean_val, std_val)
        print(f"{key}: Mean = {mean_val:.6f} ± {std_val:.6f}")

    # Per-component summary
    comp_names = concentration_columns
    comp_summary = {comp: {'RMSE': [], 'MAE': [], 'R2': []} for comp in comp_names}
    for m in all_fold_metrics:
        for comp_m in m['component_metrics']:
            comp = comp_m['Component']
            comp_summary[comp]['RMSE'].append(comp_m['RMSE'])
            comp_summary[comp]['MAE'].append(comp_m['MAE'])
            comp_summary[comp]['R2'].append(comp_m['R2'])

    summary_rows = []
    for comp in comp_names:
        rmse_mean = np.mean(comp_summary[comp]['RMSE'])
        rmse_std = np.std(comp_summary[comp]['RMSE'], ddof=1)
        r2_mean = np.mean(comp_summary[comp]['R2'])
        r2_std = np.std(comp_summary[comp]['R2'], ddof=1)
        summary_rows.append({
            'Component': comp,
            'RMSE_mean': rmse_mean,
            'RMSE_std': rmse_std,
            'R2_mean': r2_mean,
            'R2_std': r2_std
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(result_root, 'CV_summary.csv'), index=False, encoding='utf-8-sig')

    with open(os.path.join(result_root, 'CV_summary.txt'), 'w') as f:
        f.write("6-fold Cross Validation Results\n")
        f.write("="*60 + "\n")
        for key, (mean_val, std_val) in summary_stats.items():
            f.write(f"{key}: {mean_val:.6f} ± {std_val:.6f}\n")
        f.write("\nPer-component:\n")
        for row in summary_rows:
            f.write(f"{row['Component']}: RMSE={row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}, "
                    f"R2={row['R2_mean']:.4f}±{row['R2_std']:.4f}\n")

    print(f"\nAll results saved in: {os.path.abspath(result_root)}")
    print("="*70)

if __name__ == "__main__":
    main()