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
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------- Global Parameters --------------------------
folder_path = r'Mixed-gas_dataset'
concentration_file = 'Mixing Ratio.xlsx'          # relative path, will be combined with folder_path
total_classes = 100                               # total number of classes
selected_classes_num = 100                        # number of classes to sample for visualization (can be changed)
seed_value = 41
select_channel = [0, 1, 2, 3, 4, 5, 6, 7]
epochs = 1500
batch_size = 64
learning_rate = 3e-4
weight_decay = 1e-4
latent_dim = 256
augment_factor = 3
gaussian_noise_std = 0.1
time_stretch_range = (0.9, 1.1)
smooth_sigma = 1.0

RESULT_ROOT = 'Mixed-gas_quantification_results'   # root directory for saving all results

# -------------------------- Utility Functions --------------------------
def extract_class_prefix(file_name):
    match = re.match(r'^(\d+)_', file_name)
    if not match:
        raise ValueError(f"File name {file_name} does not match expected pattern")
    return match.group(1)

def extract_file_index(file_name):
    # New naming: e.g., "1_1.xlsx" -> extract trailing number after underscore
    match = re.search(r'_(\d+)\.xlsx$', file_name)
    return int(match.group(1)) if match else 0

def extract_time_domain_features_torch(seq):
    """Extract time-domain statistical features using PyTorch operations (differentiable)"""
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

# -------------------------- Data Augmentation Dataset (offline) --------------------------
class RawAugmentedDataset(Dataset):
    """Custom dataset for offline augmentation of raw data"""

    def __init__(self, data, targets, augment=True):
        self.data = data          # shape: (N, T, C) raw data (not standardized)
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

# -------------------------- Model Definition --------------------------
class CNNLSTMRegressor(nn.Module):
    """CNN + LSTM with attention for regression"""

    def __init__(self, in_channel, seq_len, output_dim, latent_dim=256):
        super(CNNLSTMRegressor, self).__init__()
        self.in_channel = in_channel
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # CNN for local features
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

        # LSTM for temporal dependencies
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

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * 2, 1),
            nn.Softmax(dim=1)
        )

        # Time-domain statistic features
        self.statistic_fc = nn.Sequential(
            nn.Linear(len(select_channel) * 6, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        # Regression head
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
        # x: (B, T, C) -> (B, C, T)
        x_cnn = x.permute(0, 2, 1)[:, select_channel, :]

        cnn_feat = self.cnn(x_cnn)               # (B, 256, T//4)
        cnn_feat = cnn_feat.permute(0, 2, 1)     # (B, T//4, 256)

        lstm_feat, _ = self.lstm(cnn_feat)       # (B, T//4, 2*latent_dim)

        attn_weights = self.attention(lstm_feat) # (B, T//4, 1)
        lstm_attn = torch.sum(lstm_feat * attn_weights, dim=1)  # (B, 2*latent_dim)

        batch_size = x.size(0)
        stat_features = []
        for i in range(batch_size):
            stat_feat = extract_time_domain_features_torch(x[i])
            stat_features.append(stat_feat)
        stat_features = torch.stack(stat_features)   # (B, len(select_channel)*6)
        stat_feat = self.statistic_fc(stat_features) # (B, 128)

        fused_feat = torch.cat([lstm_attn, stat_feat], dim=1)
        pred = self.regressor(fused_feat)
        return pred

# -------------------------- Training and Evaluation Functions --------------------------
def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    squared_error = 0.5 * error ** 2
    absolute_error = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(torch.abs(error) <= delta, squared_error, absolute_error).mean()

def train_model(model, train_loader, device, epochs, criterion,
                optimizer, scheduler, result_dir):
    train_losses = []
    best_train_loss = float('inf')
    patience = 50
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_total += batch_size

        avg_train_loss = train_loss / train_total
        train_losses.append(avg_train_loss)

        scheduler.step(avg_train_loss)

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_model_params.pth'))
            counter = 0
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | Model saved | "
                  f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        else:
            counter += 1
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        if counter >= patience:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | Early stopping triggered")
            break

    pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses
    }).to_csv(os.path.join(result_dir, 'train_process_data.csv'), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'train_loss_curve.png'), dpi=300)
    plt.close()

    return best_train_loss

def evaluate_model(model, test_loader, device, concentration_columns, scaler_y, result_dir,
                   selected_prefixes, full_labels, test_indices):
    model.eval()
    all_preds = []
    all_targets = []
    all_class_labels = []
    test_labels = [full_labels[i] for i in test_indices]

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            start_idx = idx * batch_size
            end_idx = start_idx + len(inputs)
            all_class_labels.extend(test_labels[start_idx:end_idx])

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_preds_original = scaler_y.inverse_transform(all_preds)
    all_targets_original = scaler_y.inverse_transform(all_targets)
    all_residuals = all_preds_original - all_targets_original

    residuals_df = pd.DataFrame()
    residuals_df['Sample Index'] = np.repeat(test_indices, len(concentration_columns))
    residuals_df['Component'] = np.tile(concentration_columns, len(test_indices))
    residuals_df['Residual'] = all_residuals.flatten()
    residuals_df.to_excel(os.path.join(result_dir, 'residual_distribution_data.xlsx'), index=False)

    overall_residuals = all_residuals.flatten()

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    component_metrics = []
    for i, component in enumerate(concentration_columns):
        comp_mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
        component_metrics.append({
            'Component': component,
            'MSE': comp_mse,
            'RMSE': np.sqrt(comp_mse),
            'MAE': mean_absolute_error(all_targets[:, i], all_preds[:, i]),
            'R²': r2_score(all_targets[:, i], all_preds[:, i])
        })

    with open(os.path.join(result_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Test Set Evaluation Metrics\n=============================\n")
        f.write(f"MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}\n\n")
        for metric in component_metrics:
            f.write(f"{metric['Component']}: RMSE={metric['RMSE']:.4f}, R²={metric['R²']:.4f}\n")

    pd.DataFrame(component_metrics).to_csv(
        os.path.join(result_dir, 'component_metrics.csv'), index=False, encoding='utf-8-sig'
    )

    pred_df = pd.DataFrame({'Class': all_class_labels})
    for i, component in enumerate(concentration_columns):
        pred_df[f'{component}_Actual'] = all_targets_original[:, i]
        pred_df[f'{component}_Predicted'] = all_preds_original[:, i]
        pred_df[f'{component}_Residual'] = all_residuals[:, i]
    pred_df.to_csv(os.path.join(result_dir, 'predictions_vs_actuals.csv'), index=False, encoding='utf-8-sig')

    # Residual distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(overall_residuals, kde=True, bins=50, color='#1f77b4')
    plt.axvline(0, color='red', linestyle='--', label='Zero residual')
    plt.xlabel('Residual (Predicted - Actual) (ppb)')
    plt.ylabel('Frequency')
    plt.title(f'Overall Residual Distribution (RMSE={rmse:.2f} ppb)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(result_dir, 'residual_distribution.png'), dpi=300)
    plt.close()

    # Scatter plots per component
    n_components = len(concentration_columns)
    n_rows = (n_components + 1) // 2
    plt.figure(figsize=(14, 5 * n_rows))
    for i, component in enumerate(concentration_columns):
        plt.subplot(n_rows, 2, i + 1)
        plt.scatter(all_targets_original[:, i], all_preds_original[:, i], alpha=0.6, s=30)
        min_val = min(all_targets_original[:, i].min(), all_preds_original[:, i].min())
        max_val = max(all_targets_original[:, i].max(), all_preds_original[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel(f'Actual {component} (ppb)')
        plt.ylabel(f'Predicted {component} (ppb)')
        plt.title(f'{component} (R²={component_metrics[i]["R²"]:.4f})')
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    # Per-class summary
    class_data_dir = os.path.join(result_dir, 'class_data')
    os.makedirs(class_data_dir, exist_ok=True)

    unique_classes = sorted(list(set(all_class_labels)))
    np.random.seed(seed_value)
    selected_classes = np.random.choice(unique_classes, min(selected_classes_num, len(unique_classes)), replace=False)

    all_class_data = pd.DataFrame()

    for cls in selected_classes:
        cls_indices = [i for i, label in enumerate(all_class_labels) if label == cls]
        cls_preds = all_preds_original[cls_indices]
        cls_targets = all_targets_original[cls_indices]
        cls_residuals = all_residuals[cls_indices]

        class_data = pd.DataFrame()
        class_data['Class'] = [cls] * len(cls_indices)
        class_data['Sample Index'] = [test_indices[i] for i in cls_indices]

        for i, component in enumerate(concentration_columns):
            class_data[f'{component}_Actual'] = cls_targets[:, i]
            class_data[f'{component}_Predicted'] = cls_preds[:, i]
            class_data[f'{component}_Residual'] = cls_residuals[:, i]

        all_class_data = pd.concat([all_class_data, class_data], ignore_index=True)

    all_class_data.to_excel(os.path.join(class_data_dir, 'selected_classes_summary.xlsx'), index=False)

    for cls in selected_classes:
        cls_indices = [i for i, label in enumerate(all_class_labels) if label == cls]
        cls_preds = all_preds_original[cls_indices]
        cls_targets = all_targets_original[cls_indices]

        avg_preds = np.mean(cls_preds, axis=0)
        avg_targets = np.mean(cls_targets, axis=0)

        x = np.arange(len(concentration_columns))
        width = 0.35

        plt.figure(figsize=(12, 6))
        bars1 = plt.bar(x - width / 2, avg_targets, width, label='Actual')
        bars2 = plt.bar(x + width / 2, avg_preds, width, label='Predicted')

        plt.xlabel('Gas Component')
        plt.ylabel('Concentration (ppb)')
        plt.title(f'Class {cls} - Concentration Prediction Comparison')
        plt.xticks(x, concentration_columns, rotation=45, ha='right')
        plt.legend()

        def add_bar_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', rotation=0)

        add_bar_labels(bars1)
        add_bar_labels(bars2)

        plt.tight_layout()
        plt.savefig(os.path.join(class_data_dir, f'class_{cls}_comparison.png'), dpi=300)
        plt.close()

    return mse, rmse, mae, r2

# -------------------------- Main Program --------------------------
def main():
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print(f"Loading data... Total classes: {total_classes}, Selected classes for visualization: {selected_classes_num}")
    print("Data pipeline: raw data + augmented data -> standardization (fit on augmented training set)")
    print("=" * 60)

    # Load concentration file (located in folder_path)
    conc_file_path = os.path.join(folder_path, concentration_file)
    concentration_df = pd.read_excel(conc_file_path)
    concentration_columns = concentration_df.columns[1:].tolist()
    print(f"Components: {concentration_columns}")

    # Load sample files, EXCLUDING the concentration file itself
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and f != concentration_file]
    class_file_dict = {}
    for f in file_list:
        prefix = extract_class_prefix(f)
        class_file_dict.setdefault(prefix, []).append(f)

    sorted_prefixes = sorted(class_file_dict.keys(), key=int)
    if len(sorted_prefixes) != total_classes:
        raise ValueError(f"Expected {total_classes} classes, found {len(sorted_prefixes)}")

    full_data = []
    full_concentrations = []
    full_labels = []
    class_sample_indices = {}
    current_idx = 0

    for prefix in sorted_prefixes:
        class_files = sorted(class_file_dict[prefix], key=extract_file_index)

        if len(class_files) < 6:
            raise ValueError(f"Class {prefix} has only {len(class_files)} samples (<6)")

        conc_row = concentration_df[concentration_df['No.'] == int(prefix)]
        if conc_row.empty:
            raise ValueError(f"Concentration record for class {prefix} not found")
        class_conc = conc_row.iloc[0, 1:].values

        class_sample_indices[prefix] = (current_idx, current_idx + len(class_files))

        for f in class_files:
            df = pd.read_excel(os.path.join(folder_path, f))
            data_cols = df.iloc[:, 1:].values
            full_data.append(data_cols)
            full_concentrations.append(class_conc)
            full_labels.append(prefix)
            current_idx += 1

    min_rows = min(len(s) for s in full_data)
    full_data = [s[:min_rows] for s in full_data]
    full_data = np.array(full_data)
    full_concentrations = np.array(full_concentrations)
    print(f"Raw data loaded: {full_data.shape} (samples × time steps × channels)")

    # Train/test split (5:1 per class)
    train_indices = []
    test_indices = []
    for prefix in sorted_prefixes:
        start, end = class_sample_indices[prefix]
        indices = list(range(start, end))
        np.random.shuffle(indices)
        train_indices.extend(indices[:5])
        test_indices.append(indices[5])

    X_train_raw = full_data[train_indices]
    y_train_raw = full_concentrations[train_indices]
    X_test_raw = full_data[test_indices]
    y_test_raw = full_concentrations[test_indices]
    print(f"Split: training {X_train_raw.shape[0]} samples, test {X_test_raw.shape[0]} samples")

    # Offline augmentation for training set (original + augmented)
    print(f"\nGenerating augmented training data (augment factor: {augment_factor})")
    train_aug_data = X_train_raw.tolist()
    train_aug_targets = y_train_raw.tolist()

    train_aug_dataset = RawAugmentedDataset(X_train_raw, y_train_raw, augment=True)
    for idx in range(len(train_aug_dataset)):
        data, target = train_aug_dataset[idx]
        train_aug_data.append(data)
        train_aug_targets.append(target)

    train_aug_data = np.array(train_aug_data)
    train_aug_targets = np.array(train_aug_targets)
    print(f"Augmentation done: original {len(X_train_raw)} + augmented {len(train_aug_dataset)} = total {len(train_aug_data)} samples")

    # Standardization (fit on augmented training data)
    print("\nStandardizing (fit on augmented training set)")
    scaler_x = StandardScaler()
    train_shape = train_aug_data.shape
    X_train_scaled = scaler_x.fit_transform(
        train_aug_data.reshape(-1, train_shape[1] * train_shape[2])
    ).reshape(train_shape)

    test_shape = X_test_raw.shape
    X_test_scaled = scaler_x.transform(
        X_test_raw.reshape(-1, test_shape[1] * test_shape[2])
    ).reshape(test_shape)

    # MinMax scaling for concentrations
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(train_aug_targets)
    y_test_scaled = scaler_y.transform(y_test_raw)
    print("Standardization completed.")

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNNLSTMRegressor(
        in_channel=len(select_channel),
        seq_len=full_data.shape[1],
        output_dim=full_concentrations.shape[1],
        latent_dim=latent_dim
    ).to(device)

    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-7)

    # Create result directory (under RESULT_ROOT with timestamp)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(RESULT_ROOT, f'run_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}")

    print(f"\nStarting training for {epochs} epochs")
    train_model(
        model, train_loader, device, epochs,
        criterion=huber_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        result_dir=result_dir
    )

    print("\nEvaluating best model")
    model.load_state_dict(torch.load(os.path.join(result_dir, 'best_model_params.pth')))
    mse, rmse, mae, r2 = evaluate_model(
        model, test_loader, device, concentration_columns, scaler_y, result_dir,
        sorted_prefixes, full_labels, test_indices
    )

    print("\n" + "=" * 60)
    print(f"Regression results (selected {selected_classes_num} classes, augmented training set)")
    print("=" * 60)
    print(f"Test RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")
    print(f"Results path: {os.path.abspath(result_dir)}")

if __name__ == "__main__":
    main()