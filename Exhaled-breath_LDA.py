import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2
from matplotlib.patches import Ellipse

warnings.filterwarnings('ignore')

# ========================== GLOBAL CONFIGURATION ==========================
RAW_DATA_FOLDER = r'Exhaled breath_perturbed_dataset'   # change as needed
STAT_INFO_FILE = r'Exhaled breath statistics information.xlsx'
RESULT_ROOT = r'Exhaled breath_results'

# Select labels: 'status', 'habit', 'bmi', 'bmi diff'
SELECTED_LABELS = ['status', 'habit', 'bmi']   # add others if needed

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ========================== UTILITY FUNCTIONS ==========================
def extract_class_prefix(file_name):
    match = re.match(r'^(\d+)_', file_name)
    if not match:
        raise ValueError(f"File name {file_name} does not start with 'number_'")
    return int(match.group(1))

def extract_sample_features(sample_data):
    features = []
    for channel in range(sample_data.shape[1]):
        channel_data = sample_data[:, channel]
        features.append(np.mean(channel_data))
        features.append(np.var(channel_data))
        features.append(np.max(channel_data))
        features.append(np.min(channel_data))
        features.append(scipy.stats.kurtosis(channel_data))
        features.append(scipy.stats.skew(channel_data))
    return np.array(features)

def plot_confidence_ellipse(ax, mean, cov, confidence=0.9, color='blue', alpha=0.2, linewidth=1.5):
    """Draw a 2D confidence ellipse."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    chi2_val = chi2.ppf(confidence, df=2)
    width = 2 * np.sqrt(chi2_val * eigvals[0])
    height = 2 * np.sqrt(chi2_val * eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=alpha, linewidth=linewidth, linestyle='--')
    ax.add_patch(ellipse)
    return {'mean_x': mean[0], 'mean_y': mean[1], 'width': width, 'height': height, 'angle': angle,
            'eigval1': eigvals[0], 'eigval2': eigvals[1], 'eigvec1_x': eigvecs[0, 0], 'eigvec1_y': eigvecs[1, 0]}

# ========================== MAIN PROCESSING FUNCTION ==========================
def process_label(label):
    print("\n" + "=" * 80)
    print(f"Processing label: {label.upper()}")
    print("=" * 80)

    # ---- Build result directory based on input data folder ----
    base_name = os.path.basename(RAW_DATA_FOLDER)
    if base_name.endswith('_dataset'):
        base_name = base_name[:-8]
    folder_label = label.replace(' ', '_')
    result_root = os.path.join(RESULT_ROOT, base_name, f'{folder_label}_LDA')
    os.makedirs(result_root, exist_ok=True)

    # ---- 1. Load raw time-series data ----
    print("\n1. Loading raw time-series data")
    file_list = [f for f in os.listdir(RAW_DATA_FOLDER) if f.endswith('.xlsx')]
    if not file_list:
        raise FileNotFoundError(f"No Excel files found in {RAW_DATA_FOLDER}")

    class_file_dict = {}
    for file_name in file_list:
        cls = extract_class_prefix(file_name)
        class_file_dict.setdefault(cls, []).append(file_name)

    sorted_classes = sorted(class_file_dict.keys())
    if len(sorted_classes) != 110:
        raise ValueError(f"Expected 110 classes, got {len(sorted_classes)}")
    for cls in sorted_classes:
        if len(class_file_dict[cls]) != 6:
            raise ValueError(f"Class {cls} has {len(class_file_dict[cls])} samples, expected 6")

    all_data = []
    all_class_labels = []
    all_sample_names = []

    for cls in sorted_classes:
        def extract_file_idx(file_name):
            match = re.search(r'_(\d+)\.xlsx$', file_name)
            return int(match.group(1)) if match else 0

        class_files_sorted = sorted(class_file_dict[cls], key=extract_file_idx)
        for file_name in class_files_sorted:
            file_path = os.path.join(RAW_DATA_FOLDER, file_name)
            df = pd.read_excel(file_path).iloc[:, 1:]
            all_data.append(df.values)
            all_class_labels.append(cls)
            all_sample_names.append(file_name)

    min_time_steps = min(len(sample) for sample in all_data)
    all_data = [sample[:min_time_steps] for sample in all_data]
    all_data = np.array(all_data)
    print(f"Data loaded: {len(all_data)} samples, each shape {all_data[0].shape}")

    # ---- 2. Feature extraction ----
    print("\n2. Feature extraction")
    scaler = StandardScaler()
    data_shape = all_data.shape
    flat = all_data.reshape(-1, data_shape[1] * data_shape[2])
    scaled = scaler.fit_transform(flat).reshape(data_shape)

    all_features = []
    for sample in scaled:
        all_features.append(extract_sample_features(sample))
    all_features = np.array(all_features)
    print(f"Feature extraction done: {all_features.shape[1]} dimensions")

    # ---- 3. Label matching and encoding ----
    print(f"\n3. Loading statistical info and encoding {label} labels")
    stat_df = pd.read_excel(STAT_INFO_FILE)
    stat_df.columns = [col.strip().lower().replace(' ', '') for col in stat_df.columns]

    id_candidates = ['id', '编号', '类别id', 'classid']
    id_col = next((c for c in stat_df.columns if c in id_candidates), None)
    if id_col is None:
        raise ValueError(f"No ID column found in {STAT_INFO_FILE}. Columns: {list(stat_df.columns)}")

    stat_df['ID_int'] = stat_df[id_col].astype(str).str.zfill(3).str.lstrip('0').replace('', '0').astype(int)
    stat_df = stat_df[stat_df['ID_int'].between(1, 110)].reset_index(drop=True)
    if len(stat_df) != 110:
        raise ValueError(f"Valid IDs count {len(stat_df)}, expected 110")

    # ---- Label-specific encoding ----
    if label == 'status':
        target_col = next((c for c in stat_df.columns if c in ['status', '状态', '类别状态', 'classstatus']), None)
        if target_col is None:
            raise ValueError(f"No status column found. Columns: {list(stat_df.columns)}")
        class_to_label = dict(zip(stat_df['ID_int'], stat_df[target_col]))
        all_labels_original = [class_to_label[cls] for cls in all_class_labels]
        encoder = LabelEncoder()
        all_labels_encoded = encoder.fit_transform(all_labels_original)
        class_names = encoder.classes_
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    elif label == 'habit':
        target_col = next((c for c in stat_df.columns if c in ['habit', '习惯', '类别习惯', 'classhabit']), None)
        if target_col is None:
            raise ValueError(f"No habit column found. Columns: {list(stat_df.columns)}")
        class_to_label = dict(zip(stat_df['ID_int'], stat_df[target_col]))
        all_labels_original = [class_to_label[cls] for cls in all_class_labels]
        encoder = LabelEncoder()
        all_labels_encoded = encoder.fit_transform(all_labels_original)
        class_names = encoder.classes_
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    elif label == 'bmi':
        target_col = next((c for c in stat_df.columns if c in ['bmi', '身体质量指数']), None)
        if target_col is None:
            raise ValueError(f"No BMI column found. Columns: {list(stat_df.columns)}")
        class_to_bmi = dict(zip(stat_df['ID_int'], stat_df[target_col]))
        all_bmi_values = [class_to_bmi[cls] for cls in all_class_labels]
        all_labels_encoded = []
        for v in all_bmi_values:
            if v < 18.5:
                all_labels_encoded.append(0)
            elif 18.5 <= v <= 23.9:
                all_labels_encoded.append(1)
            elif 24 <= v <= 27.9:
                all_labels_encoded.append(2)
            else:
                all_labels_encoded.append(3)
        class_names = ['BMI < 18.5', '18.5 ≤ BMI ≤ 23.9', '24 ≤ BMI ≤ 27.9', 'BMI ≥ 28']
        mapping = dict(zip(class_names, [0, 1, 2, 3]))
        all_labels_original = all_bmi_values

    elif label == 'bmi diff':
        # ---------- Strict match for BMI difference columns ----------
        target_bmi_cols = ['bmi diff', 'bmi_diff', '差值']
        norm_targets = [col.strip().lower().replace(' ', '') for col in target_bmi_cols]
        bmi_col = None
        for col in stat_df.columns:
            if col in norm_targets:
                bmi_col = col
                break
        if bmi_col is None:
            print(f"Available columns: {list(stat_df.columns)}")
            raise ValueError(f"No BMI diff column found. Please check column names.")
        print(f"  Using column '{bmi_col}' for BMI diff values.")
        class_to_val = dict(zip(stat_df['ID_int'], stat_df[bmi_col]))
        all_vals = [class_to_val[cls] for cls in all_class_labels]
        all_labels_encoded = []
        for v in all_vals:
            if v < 0:
                all_labels_encoded.append(0)
            elif v == 0:
                all_labels_encoded.append(1)
            else:
                all_labels_encoded.append(2)
        unique_vals = sorted(set(all_labels_encoded))
        class_names = []
        for code in unique_vals:
            if code == 0:
                class_names.append('BMI_diff < 0')
            elif code == 1:
                class_names.append('BMI_diff = 0')
            elif code == 2:
                class_names.append('BMI_diff > 0')
        old_to_new = {old: new for new, old in enumerate(unique_vals)}
        all_labels_encoded = [old_to_new[val] for val in all_labels_encoded]
        mapping = dict(zip(class_names, list(range(len(class_names)))))
        all_labels_original = all_vals

    else:
        raise ValueError(f"Unsupported label: {label}")

    if len(class_names) < 2:
        raise ValueError(f"Only {len(class_names)} class(es) found. LDA requires at least 2 classes.")

    print(f"Label encoding done. Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    # ---- 4. LDA dimensionality reduction and classifier training ----
    print("\n4. LDA reduction and classifier training")
    n_classes = len(class_names)
    n_components = min(2, n_classes - 1)

    lda_reducer = LDA(n_components=n_components, solver='svd', store_covariance=True)
    all_features_lda = lda_reducer.fit_transform(all_features, all_labels_encoded)

    if all_features_lda.shape[1] == 1:
        all_features_lda = np.hstack([all_features_lda, np.zeros((all_features_lda.shape[0], 1))])

    lda_clf = LDA(solver='svd')
    lda_clf.fit(all_features_lda, all_labels_encoded)

    # ---- Save LDA reduction results ----
    lda_df = pd.DataFrame({
        'SampleName': all_sample_names,
        'ClassID': all_class_labels,
        'OriginalLabel': all_labels_original,
        'EncodedLabel': all_labels_encoded,
        'LDA_x': all_features_lda[:, 0],
        'LDA_y': all_features_lda[:, 1]
    })
    lda_df.to_csv(os.path.join(result_root, 'LDA_Reduction.csv'), index=False, encoding='utf-8-sig')

    # ---- Decision boundary grid ----
    x_min, x_max = all_features_lda[:, 0].min() - 0.5, all_features_lda[:, 0].max() + 0.5
    y_min, y_max = all_features_lda[:, 1].min() - 0.5, all_features_lda[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = lda_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    num_classes = len(class_names)
    if num_classes <= 10:
        colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    elif num_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_classes))

    # ---- Figure 1: Confidence ellipses ----
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    for i, cls_name in enumerate(class_names):
        mask = np.array(all_labels_encoded) == i
        points = all_features_lda[mask]
        if len(points) >= 2:
            mean = np.mean(points, axis=0)
            cov = np.cov(points, rowvar=False)
            plot_confidence_ellipse(ax1, mean, cov, confidence=0.9,
                                    color=colors[i], alpha=0.15, linewidth=1.5)
        ax1.scatter(points[:, 0], points[:, 1],
                    c=[colors[i]], label=f'{cls_name} (n={len(points)})',
                    s=80, edgecolors='white', linewidth=1.5, alpha=0.9)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True)
    ax1.set_xlabel('LDA Component 1', fontsize=14)
    ax1.set_ylabel('LDA Component 2', fontsize=14)
    ax1.set_title(f'LDA Reduction + 90% Confidence Ellipses (cumulative explained variance: {sum(lda_reducer.explained_variance_ratio_):.2%})',
                  fontsize=16)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(result_root, 'LDA_ConfidenceEllipse.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ---- 5. Cross-validation and permutation test ----
    print("\n5. Cross-validation for average confusion matrix percentage")
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    cv_scores = []
    conf_mat_sum = np.zeros((len(class_names), len(class_names)), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(all_features_lda, all_labels_encoded), 1):
        X_train, X_val = all_features_lda[train_idx], all_features_lda[val_idx]
        y_train, y_val = np.array(all_labels_encoded)[train_idx], np.array(all_labels_encoded)[val_idx]
        clf_cv = LDA(solver='svd')
        clf_cv.fit(X_train, y_train)
        acc = clf_cv.score(X_val, y_val)
        cv_scores.append(acc)
        y_pred_cv = clf_cv.predict(X_val)
        cm = confusion_matrix(y_val, y_pred_cv, labels=range(len(class_names)))
        conf_mat_sum += cm
        print(f"  Fold {fold_idx} accuracy: {acc:.4f}")

    cv_scores = np.array(cv_scores)
    print(f"6-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    conf_mat_avg = conf_mat_sum / cv.n_splits
    row_sum = conf_mat_avg.sum(axis=1, keepdims=True)
    avg_pct = np.divide(conf_mat_avg, row_sum, out=np.zeros_like(conf_mat_avg, dtype=float),
                        where=row_sum != 0) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_pct, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Average Confusion Matrix (6-Fold CV) - Row-wise % - {label}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(result_root, 'CV_AvgConfusionMatrix_Pct.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 5b. Permutation test (only print p-value, no extra files) ----
    print("\nPerforming permutation test for significance (p-value):")
    n_perm = 1000
    perm_scores = []
    rng = np.random.RandomState(42)
    observed_score = lda_clf.score(all_features_lda, all_labels_encoded)
    for _ in range(n_perm):
        y_perm = rng.permutation(all_labels_encoded)
        clf_perm = LDA(solver='svd')
        clf_perm.fit(all_features_lda, y_perm)
        perm_scores.append(clf_perm.score(all_features_lda, y_perm))
    perm_scores = np.array(perm_scores)
    p_value = (np.sum(perm_scores >= observed_score) + 1) / (n_perm + 1)
    print(f"  Observed accuracy: {observed_score:.4f}, random mean: {perm_scores.mean():.4f}, p-value: {p_value:.6f}")

    # ---- 6. Export boundary data ----
    print("\n6. Exporting boundary data to Excel")
    excel_path = os.path.join(result_root, 'LDA_BoundaryData.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        grid_export = pd.DataFrame({
            'GridX': xx.ravel(),
            'GridY': yy.ravel(),
            'PredictedEncoded': Z.ravel(),
            'PredictedOriginal': [class_names[i] for i in Z.ravel()]
        })
        grid_export.to_excel(writer, sheet_name='GridPoints', index=False)

        sample_export = lda_df[['SampleName', 'ClassID', 'OriginalLabel', 'EncodedLabel', 'LDA_x', 'LDA_y']].copy()
        sample_export.to_excel(writer, sheet_name='SamplePoints', index=False)

        mapping_export = pd.DataFrame({
            'OriginalLabel': list(mapping.keys()),
            'EncodedLabel': list(mapping.values())
        })
        mapping_export.to_excel(writer, sheet_name='LabelMapping', index=False)

    print(f"Boundary data exported to {excel_path}")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print(f"Processing for {label.upper()} completed.")
    print(f"Results saved in: {result_root}")
    print(f"  - LDA_Reduction.csv")
    print(f"  - LDA_ConfidenceEllipse.png")
    print(f"  - CV_AvgConfusionMatrix_Pct.png")
    print(f"  - LDA_BoundaryData.xlsx")
    print(f"Permutation p-value (printed above): {p_value:.6f}")
    print("=" * 80)

# ========================== MAIN ==========================
if __name__ == '__main__':
    for lbl in SELECTED_LABELS:
        process_label(lbl)