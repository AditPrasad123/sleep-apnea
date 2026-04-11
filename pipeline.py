import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, find_peaks, filtfilt, resample_poly
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


DATA_DIR = "apnea_data"
ARTIFACT_DIR = "artifacts"
CROSS_ARTIFACT_DIR = os.path.join(ARTIFACT_DIR, "cross-dataset-models")
FS = 100
WINDOW_SECONDS = 30
STRIDE_SECONDS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 30
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
MC_DROPOUT_PASSES = 25
SIGNAL_CHANNELS = 2
THRESHOLD_METRICS = ("f1", "balanced_accuracy", "mcc")

FEATURE_NAMES = [
    "mean",
    "std",
    "variance",
    "rms",
    "max",
    "min",
    "median",
    "p25",
    "p75",
    "peak_count",
    "mean_rr",
    "std_rr",
    "rmssd",
    "pnn50",
    "max_rr",
    "min_rr",
    "median_rr",
    "q25_rr",
    "q75_rr",
    "energy",
    "entropy",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "zero_crossings",
    "skewness",
    "kurtosis",
    "peak_std",
    "signal_range",
    "abs_mean",
]

APNEA_TOKENS = {"OA", "X", "CA", "CAA", "H", "HA"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionNet(nn.Module):
    def __init__(self, feature_dim, input_channels=SIGNAL_CHANNELS):
        super().__init__()

        self.signal_conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.signal_lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.signal_lstm_2 = nn.LSTM(
            input_size=128,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.signal_fc = nn.Linear(64, 64)
        self.signal_dropout = nn.Dropout(0.3)

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, signal, feature):
        x = self.signal_conv(signal)
        x = x.transpose(1, 2)
        x, _ = self.signal_lstm(x)
        x, _ = self.signal_lstm_2(x)
        x = x[:, -1, :]
        cnn_out = self.signal_dropout(torch.relu(self.signal_fc(x)))

        feat_out = self.feature_fc(feature)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits


class CNNBaseline(nn.Module):
    def __init__(self, input_channels=SIGNAL_CHANNELS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, signal):
        features = self.backbone(signal)
        logits = self.classifier(features).squeeze(1)
        return logits


def ensure_artifact_dir(base_dir=ARTIFACT_DIR):
    os.makedirs(base_dir, exist_ok=True)


def artifact_path(name, base_dir=ARTIFACT_DIR):
    return os.path.join(base_dir, name)


def cross_artifact_path(name):
    return artifact_path(name, base_dir=CROSS_ARTIFACT_DIR)


def save_json(data, filename, base_dir=ARTIFACT_DIR):
    ensure_artifact_dir(base_dir)
    with open(artifact_path(filename, base_dir=base_dir), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(filename, base_dir=ARTIFACT_DIR):
    with open(artifact_path(filename, base_dir=base_dir), "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_segment(segment):
    segment = np.nan_to_num(np.asarray(segment, dtype=np.float32))
    std = np.std(segment)
    if std <= 0:
        return None
    segment = (segment - np.mean(segment)) / (std + 1e-8)
    if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
        return None
    if np.max(segment) - np.min(segment) < 0.05:
        return None
    return np.asarray(segment, dtype=np.float32)


def tokenize_aux(note):
    return [token.strip().upper() for token in note.strip().split() if token.strip()]


def has_apnea_event(note):
    return any(token in APNEA_TOKENS for token in tokenize_aux(note))


def select_ecg_channel(record):
    sig_names = [name.lower() for name in record.sig_name]
    for idx, name in enumerate(sig_names):
        if "ecg" in name:
            return idx
    return 0


def resample_to_target(segment, original_fs, target_fs=FS):
    if int(round(original_fs)) == int(target_fs):
        return np.asarray(segment, dtype=np.float32)

    original_fs_int = int(round(original_fs))
    target_fs_int = int(target_fs)
    resampled = resample_poly(segment, up=target_fs_int, down=original_fs_int)

    target_len = target_fs_int * 30
    if len(resampled) > target_len:
        resampled = resampled[:target_len]
    elif len(resampled) < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[: len(resampled)] = resampled
        resampled = padded

    return np.asarray(resampled, dtype=np.float32)


def bandpass_filter_segment(segment, fs=FS, lowcut=0.5, highcut=40.0, order=4):
    segment = np.asarray(segment, dtype=np.float32)
    if len(segment) < max(10, order * 6):
        return segment

    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 1e-4)
    high = min(highcut / nyquist, 0.999)
    if low >= high:
        return segment

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, segment).astype(np.float32)


def harmonize_ecg_segment_full(segment, fs=FS, lowcut=0.5, highcut=40.0, clip_percentiles=(1.0, 99.0), z_clip=5.0):
    segment = np.nan_to_num(np.asarray(segment, dtype=np.float32))
    if segment.size == 0:
        return None

    segment = bandpass_filter_segment(segment, fs=fs, lowcut=lowcut, highcut=highcut)

    if clip_percentiles is not None:
        low_p, high_p = np.percentile(segment, clip_percentiles)
        segment = np.clip(segment, low_p, high_p)

    mean = np.mean(segment)
    std = np.std(segment)
    if std <= 0:
        return None

    segment = (segment - mean) / (std + 1e-8)
    if z_clip is not None:
        segment = np.clip(segment, -z_clip, z_clip)

    if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
        return None

    return np.asarray(segment, dtype=np.float32)


def harmonize_ecg_segment_light(segment, clip_percentiles=(0.5, 99.5), z_clip=None):
    segment = np.nan_to_num(np.asarray(segment, dtype=np.float32))
    if segment.size == 0:
        return None

    if clip_percentiles is not None:
        low_p, high_p = np.percentile(segment, clip_percentiles)
        segment = np.clip(segment, low_p, high_p)

    std = np.std(segment)
    if std <= 0:
        return None

    segment = (segment - np.mean(segment)) / (std + 1e-8)
    if z_clip is not None:
        segment = np.clip(segment, -z_clip, z_clip)

    if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
        return None

    return np.asarray(segment, dtype=np.float32)


def preprocess_ecg_segment(segment, fs=FS, harmonize_level="none"):
    if harmonize_level == "none":
        return normalize_segment(segment)
    if harmonize_level == "light":
        return harmonize_ecg_segment_light(segment)
    if harmonize_level == "full":
        return harmonize_ecg_segment_full(segment, fs=fs)
    raise ValueError(f"Unsupported harmonize_level: {harmonize_level}")


def load_apnea_ecg_segments_30s(data_dir=DATA_DIR, fs=FS, harmonize_level="none"):
    window = fs * 30
    minute = fs * 60

    segments = []
    labels = []

    records = sorted(name[:-4] for name in os.listdir(data_dir) if name.endswith(".dat"))

    for rec in records:
        try:
            ann = wfdb.rdann(os.path.join(data_dir, rec), "apn")
            record = wfdb.rdrecord(os.path.join(data_dir, rec))
        except Exception:
            continue

        signal = record.p_signal[:, 0]
        minute_labels = ann.symbol

        for start in range(0, len(signal) - window + 1, window):
            end = start + window
            minute_idx = start // minute
            if minute_idx >= len(minute_labels):
                continue

            segment = preprocess_ecg_segment(signal[start:end], fs=fs, harmonize_level=harmonize_level)
            if segment is None:
                continue

            label = 1 if minute_labels[minute_idx] == "A" else 0
            segments.append(segment)
            labels.append(label)

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_mitbih_psg_segments_30s(data_dir="mitbih_psg_data", target_fs=FS, harmonize_level="none"):
    segments = []
    labels = []

    records = sorted(name[:-4] for name in os.listdir(data_dir) if name.endswith(".hea"))

    for rec in records:
        try:
            record = wfdb.rdrecord(os.path.join(data_dir, rec))
            ann = wfdb.rdann(os.path.join(data_dir, rec), "st")
        except Exception:
            continue

        ecg_idx = select_ecg_channel(record)
        ecg = record.p_signal[:, ecg_idx]
        record_fs = float(record.fs)
        window = int(round(30 * record_fs))

        for i, start in enumerate(ann.sample):
            start = int(start)
            end = start + window
            if end > len(ecg):
                continue

            raw_segment = ecg[start:end]
            resampled = resample_to_target(raw_segment, original_fs=record_fs, target_fs=target_fs)
            segment = preprocess_ecg_segment(resampled, fs=target_fs, harmonize_level=harmonize_level)
            if segment is None:
                continue

            note = ann.aux_note[i] if i < len(ann.aux_note) else ""
            label = 1 if has_apnea_event(note) else 0
            segments.append(segment)
            labels.append(label)

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)


def summarize_split(name, y):
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    print(f"{name}: n={len(y)}, apnea={positives}, normal={negatives}")


def compute_cross_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-8)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(auc),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "specificity": float(specificity),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def tune_cross_threshold(y_true, y_prob, metric_name):
    metric_key = {
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "mcc": "mcc",
    }[metric_name]

    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        metrics = compute_cross_metrics(y_true, y_prob, threshold=float(threshold))
        score = metrics[metric_key]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def compare_threshold_metrics(y_true, y_prob, metric_names=THRESHOLD_METRICS):
    comparison = {}
    for metric_name in metric_names:
        threshold, score = tune_cross_threshold(y_true, y_prob, metric_name)
        comparison[metric_name] = {
            "threshold": float(threshold),
            "val_score": float(score),
        }
    return comparison


def build_signal_view(x_signal, signal_mode, fs=FS):
    if signal_mode == "ecg_edr":
        signal_arr = build_ecg_edr_signal(x_signal, fs=fs)
        return signal_arr, ["ecg", "edr"]

    if signal_mode == "ecg":
        signal_arr = np.asarray(x_signal, dtype=np.float32)[..., np.newaxis]
        return signal_arr, ["ecg"]

    if signal_mode == "edr":
        edr_signal = np.array([derive_edr_from_ecg(seg, fs=fs) for seg in np.asarray(x_signal)], dtype=np.float32)
        signal_arr = edr_signal[..., np.newaxis]
        return signal_arr, ["edr"]

    raise ValueError(f"Unsupported signal_mode: {signal_mode}")


def extract_features(segment, fs=FS):
    features = []
    segment = np.nan_to_num(segment)

    features.append(np.mean(segment))
    features.append(np.std(segment))
    features.append(np.var(segment))
    features.append(np.sqrt(np.mean(segment**2)))
    features.append(np.max(segment))
    features.append(np.min(segment))
    features.append(np.median(segment))
    features.append(np.percentile(segment, 25))
    features.append(np.percentile(segment, 75))

    peaks, _ = find_peaks(segment, distance=fs * 0.3)

    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / fs

        hr = len(peaks)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)

        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0

        nn50 = np.sum(np.abs(diff_rr) > 0.05)
        pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0

        max_rr = np.max(rr_intervals)
        min_rr = np.min(rr_intervals)
        median_rr = np.median(rr_intervals)
        q25_rr = np.percentile(rr_intervals, 25)
        q75_rr = np.percentile(rr_intervals, 75)
    else:
        hr, mean_rr, std_rr, rmssd, pnn50 = 0, 0, 0, 0, 0
        max_rr, min_rr, median_rr, q25_rr, q75_rr = 0, 0, 0, 0, 0

    features.extend(
        [
            hr,
            mean_rr,
            std_rr,
            rmssd,
            pnn50,
            max_rr,
            min_rr,
            median_rr,
            q25_rr,
            q75_rr,
        ]
    )

    energy = np.sum(segment**2)
    features.append(energy)

    segment_clean = segment[np.isfinite(segment)]
    if len(segment_clean) > 0:
        hist, _ = np.histogram(segment_clean, bins=50)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
            entropy = -np.sum(hist * np.log(hist + 1e-8))
        else:
            entropy = 0
    else:
        entropy = 0
    features.append(entropy)

    fft_vals = np.abs(np.fft.fft(segment))
    freqs = np.fft.fftfreq(len(segment), d=1 / fs)
    mask = freqs > 0
    fft_vals = fft_vals[mask]
    freqs = freqs[mask]

    lf = np.sum(fft_vals[(freqs >= 0.04) & (freqs < 0.15)])
    hf = np.sum(fft_vals[(freqs >= 0.15) & (freqs < 0.4)])
    lf_hf_ratio = lf / (hf + 1e-8)
    features.extend([lf, hf, lf_hf_ratio])

    zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
    features.append(zero_crossings)
    features.append(skew(segment))
    features.append(kurtosis(segment))

    features.append(np.std(segment[peaks]) if len(peaks) > 2 else 0)
    features.append(np.max(segment) - np.min(segment))
    features.append(np.mean(np.abs(segment)))

    return features


def derive_edr_from_ecg(segment, fs=FS):
    segment = np.nan_to_num(np.asarray(segment, dtype=np.float32))
    if segment.size == 0:
        return segment

    peaks, _ = find_peaks(segment, distance=max(1, int(0.3 * fs)))

    if len(peaks) >= 2:
        # Use beat-to-beat R-peak amplitude modulation as a respiration surrogate.
        r_amplitudes = segment[peaks]
        edr = np.interp(np.arange(len(segment)), peaks, r_amplitudes)
    else:
        # Fallback to a slow trend when R-peak extraction is unreliable.
        smooth_window = max(3, int(1.0 * fs))
        kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        edr = np.convolve(segment, kernel, mode="same")

    post_window = max(3, int(0.5 * fs))
    post_kernel = np.ones(post_window, dtype=np.float32) / float(post_window)
    edr = np.convolve(edr, post_kernel, mode="same")

    edr_std = np.std(edr)
    if edr_std > 0:
        edr = (edr - np.mean(edr)) / (edr_std + 1e-8)
    else:
        edr = np.zeros_like(segment, dtype=np.float32)

    return np.asarray(edr, dtype=np.float32)


def build_ecg_edr_signal(x_signal, fs=FS):
    ecg_signal = np.asarray(x_signal, dtype=np.float32)
    edr_signal = np.array([derive_edr_from_ecg(seg, fs=fs) for seg in ecg_signal], dtype=np.float32)
    return np.stack([ecg_signal, edr_signal], axis=2)


def load_segments_and_labels(data_dir, fs=FS, window_seconds=WINDOW_SECONDS, stride_seconds=STRIDE_SECONDS):
    window_size = fs * window_seconds
    stride = fs * stride_seconds

    all_segments = []
    all_labels = []

    records = sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".dat"))

    for rec in records:
        try:
            annotation = wfdb.rdann(f"{data_dir}/{rec}", "apn")
        except Exception:
            continue

        record = wfdb.rdrecord(f"{data_dir}/{rec}")
        signal = record.p_signal[:, 0]
        labels = annotation.symbol

        for start in range(0, len(signal) - window_size, stride):
            end = start + window_size
            segment = signal[start:end]

            if len(segment) != window_size:
                continue
            if np.std(segment) == 0:
                continue

            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

            if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
                continue
            if np.max(segment) - np.min(segment) < 0.05:
                continue

            center = (start + end) // 2
            label_idx = center // window_size
            if label_idx >= len(labels):
                continue

            all_segments.append(segment)
            all_labels.append(1 if labels[label_idx] == "A" else 0)

    return np.array(all_segments), np.array(all_labels)


def build_feature_matrix(x_signal):
    feature_data = [extract_features(seg) for seg in x_signal]
    x_features = np.array(feature_data)
    return np.nan_to_num(x_features)


def compute_class_weights(y_train):
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale = (neg / pos) if pos > 0 else 1.0
    pos_weight = torch.tensor([scale], dtype=torch.float32).to(DEVICE)
    return scale, pos_weight


def build_train_test_split(y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return train_idx, test_idx


def train_xgboost(x_train_f, y_train, x_test_f, y_test, scale):
    model = XGBClassifier(
        n_estimators=1200,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale,
        eval_metric="logloss",
    )

    model.fit(x_train_f, y_train)
    preds = model.predict(x_test_f)
    xgb_train_prob = model.predict_proba(x_train_f)[:, 1]
    xgb_test_prob = model.predict_proba(x_test_f)[:, 1]

    print("XGBoost Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model, xgb_train_prob, xgb_test_prob


def train_fusion_model(
    x_signal_train,
    x_signal_test,
    x_feat_train,
    x_feat_test,
    y_train,
    y_test,
    pos_weight,
    feature_dim,
    input_channels=SIGNAL_CHANNELS,
):
    x_signal_train_t = torch.tensor(x_signal_train.transpose(0, 2, 1), dtype=torch.float32)
    x_signal_test_t = torch.tensor(x_signal_test.transpose(0, 2, 1), dtype=torch.float32)
    x_feat_train_t = torch.tensor(x_feat_train, dtype=torch.float32)
    x_feat_test_t = torch.tensor(x_feat_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_signal_train_t, x_feat_train_t, y_train_t)
    test_dataset = TensorDataset(x_signal_test_t, x_feat_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    fusion_model = FusionNet(feature_dim=feature_dim, input_channels=input_channels).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.0002)

    for epoch in range(EPOCHS):
        fusion_model.train()
        running_loss = 0.0

        for sig_batch, feat_batch, y_batch in train_loader:
            sig_batch = sig_batch.to(DEVICE)
            feat_batch = feat_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = fusion_model(sig_batch, feat_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sig_batch.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        fusion_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for sig_batch, feat_batch, y_batch in test_loader:
                sig_batch = sig_batch.to(DEVICE)
                feat_batch = feat_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = fusion_model(sig_batch, feat_batch)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

    return fusion_model


def train_cnn_baseline(
    x_signal_train,
    x_signal_test,
    y_train,
    y_test,
    pos_weight,
    input_channels=SIGNAL_CHANNELS,
):
    x_signal_train_t = torch.tensor(x_signal_train.transpose(0, 2, 1), dtype=torch.float32)
    x_signal_test_t = torch.tensor(x_signal_test.transpose(0, 2, 1), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_signal_train_t, y_train_t)
    test_dataset = TensorDataset(x_signal_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = CNNBaseline(input_channels=input_channels).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for sig_batch, y_batch in train_loader:
            sig_batch = sig_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(sig_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sig_batch.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sig_batch, y_batch in test_loader:
                sig_batch = sig_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                logits = model(sig_batch)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"CNN Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

    return model


def predict_probs(model, signal_arr, feature_arr, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    sig_t = torch.tensor(signal_arr.transpose(0, 2, 1), dtype=torch.float32)
    feat_t = torch.tensor(feature_arr, dtype=torch.float32)
    dummy_y = torch.zeros(len(sig_t), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(sig_t, feat_t, dummy_y),
        batch_size=batch_size,
        shuffle=False,
    )
    all_probs = []

    with torch.no_grad():
        for sig_batch, feat_batch, _ in loader:
            sig_batch = sig_batch.to(DEVICE)
            feat_batch = feat_batch.to(DEVICE)
            logits = model(sig_batch, feat_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


def predict_probs_signal(model, signal_arr, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    sig_t = torch.tensor(signal_arr.transpose(0, 2, 1), dtype=torch.float32)
    dummy_y = torch.zeros(len(sig_t), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(sig_t, dummy_y),
        batch_size=batch_size,
        shuffle=False,
    )
    all_probs = []

    with torch.no_grad():
        for sig_batch, _ in loader:
            sig_batch = sig_batch.to(DEVICE)
            logits = model(sig_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def predict_probs_mc_dropout(model, signal_arr, feature_arr, mc_passes=MC_DROPOUT_PASSES, batch_size=EVAL_BATCH_SIZE):
    sig_t = torch.tensor(signal_arr.transpose(0, 2, 1), dtype=torch.float32)
    feat_t = torch.tensor(feature_arr, dtype=torch.float32)
    dummy_y = torch.zeros(len(sig_t), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(sig_t, feat_t, dummy_y),
        batch_size=batch_size,
        shuffle=False,
    )

    all_pass_means = []

    for _ in range(mc_passes):
        enable_mc_dropout(model)
        pass_probs = []

        with torch.no_grad():
            for sig_batch, feat_batch, _ in loader:
                sig_batch = sig_batch.to(DEVICE)
                feat_batch = feat_batch.to(DEVICE)
                logits = model(sig_batch, feat_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                pass_probs.append(probs)

        pass_probs = np.concatenate(pass_probs)
        all_pass_means.append(pass_probs)

    stacked_probs = np.stack(all_pass_means, axis=0)
    mean_probs = np.mean(stacked_probs, axis=0)
    std_probs = np.std(stacked_probs, axis=0)
    return mean_probs, std_probs, stacked_probs


def predict_probs_signal_mc_dropout(model, signal_arr, mc_passes=MC_DROPOUT_PASSES, batch_size=EVAL_BATCH_SIZE):
    sig_t = torch.tensor(signal_arr.transpose(0, 2, 1), dtype=torch.float32)
    dummy_y = torch.zeros(len(sig_t), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(sig_t, dummy_y),
        batch_size=batch_size,
        shuffle=False,
    )

    all_pass_probs = []

    for _ in range(mc_passes):
        enable_mc_dropout(model)
        pass_probs = []

        with torch.no_grad():
            for sig_batch, _ in loader:
                sig_batch = sig_batch.to(DEVICE)
                logits = model(sig_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                pass_probs.append(probs)

        pass_probs = np.concatenate(pass_probs)
        all_pass_probs.append(pass_probs)

    stacked_probs = np.stack(all_pass_probs, axis=0)
    mean_probs = np.mean(stacked_probs, axis=0)
    std_probs = np.std(stacked_probs, axis=0)
    return mean_probs, std_probs, stacked_probs


def compute_signal_saliency(model, signal_sample, feature_sample):
    model.eval()

    signal_tensor = torch.tensor(signal_sample, dtype=torch.float32, device=DEVICE)
    feature_tensor = torch.tensor(feature_sample, dtype=torch.float32, device=DEVICE)

    if signal_tensor.ndim == 1:
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
    elif signal_tensor.ndim == 2:
        signal_tensor = signal_tensor.unsqueeze(0)

    if feature_tensor.ndim == 1:
        feature_tensor = feature_tensor.unsqueeze(0)

    signal_tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    # CuDNN LSTM backward requires training mode; disable CuDNN just for saliency backward.
    with torch.backends.cudnn.flags(enabled=False):
        logits = model(signal_tensor, feature_tensor)
        probability = torch.sigmoid(logits)[0]
        probability.backward()

    saliency = signal_tensor.grad.detach().abs().cpu().numpy()[0]
    if saliency.ndim == 1:
        saliency = saliency[np.newaxis, :]

    for channel in range(saliency.shape[0]):
        channel_max = saliency[channel].max()
        saliency[channel] = saliency[channel] / (channel_max + 1e-8)

    return saliency, float(probability.detach().cpu().item())


def compute_signal_saliency_signal_only(model, signal_sample):
    model.eval()

    signal_tensor = torch.tensor(signal_sample, dtype=torch.float32, device=DEVICE)

    if signal_tensor.ndim == 1:
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
    elif signal_tensor.ndim == 2:
        signal_tensor = signal_tensor.unsqueeze(0)

    signal_tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    logits = model(signal_tensor)
    probability = torch.sigmoid(logits)[0]
    probability.backward()

    saliency = signal_tensor.grad.detach().abs().cpu().numpy()[0]
    if saliency.ndim == 1:
        saliency = saliency[np.newaxis, :]

    for channel in range(saliency.shape[0]):
        channel_max = saliency[channel].max()
        saliency[channel] = saliency[channel] / (channel_max + 1e-8)

    return saliency, float(probability.detach().cpu().item())


def train_stacking(xgb_train_prob, xgb_test_prob, cnn_train_prob, cnn_test_prob, y_train, y_test):
    meta_train = np.column_stack([xgb_train_prob, cnn_train_prob])
    meta_test = np.column_stack([xgb_test_prob, cnn_test_prob])

    meta_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )

    meta_model.fit(meta_train, y_train)
    final_pred = meta_model.predict(meta_test)
    print("Stacked_accuracy:", accuracy_score(y_test, final_pred))
    return meta_model


def save_array(name, array, base_dir=ARTIFACT_DIR):
    ensure_artifact_dir(base_dir)
    np.save(artifact_path(name, base_dir=base_dir), array)


def load_array(name, base_dir=ARTIFACT_DIR):
    return np.load(artifact_path(name, base_dir=base_dir), allow_pickle=False)


def save_artifacts(scaler, xgb_model, cnn_model, fusion_model, meta_model, train_idx, test_idx, feature_dim):
    ensure_artifact_dir()
    joblib.dump(scaler, artifact_path("scaler.joblib"))
    joblib.dump(xgb_model, artifact_path("xgb_model.joblib"))
    torch.save(cnn_model.state_dict(), artifact_path("cnn_model.pt"))
    torch.save(fusion_model.state_dict(), artifact_path("fusion_model.pt"))
    joblib.dump(meta_model, artifact_path("stacking_model.joblib"))
    save_array("train_idx.npy", train_idx)
    save_array("test_idx.npy", test_idx)
    save_json(
        {
            "data_dir": DATA_DIR,
            "fs": FS,
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "feature_dim": int(feature_dim),
            "feature_names": FEATURE_NAMES,
            "signal_channels": SIGNAL_CHANNELS,
            "signal_streams": ["ecg", "edr"],
        },
        "metadata.json",
    )


def load_artifacts():
    scaler = joblib.load(artifact_path("scaler.joblib"))
    xgb_model = joblib.load(artifact_path("xgb_model.joblib"))
    meta_model = joblib.load(artifact_path("stacking_model.joblib"))
    metadata = load_json("metadata.json")
    train_idx = load_array("train_idx.npy")
    test_idx = load_array("test_idx.npy")
    signal_channels = int(metadata.get("signal_channels", 1))

    cnn_model = CNNBaseline(input_channels=signal_channels).to(DEVICE)
    cnn_model.load_state_dict(torch.load(artifact_path("cnn_model.pt"), map_location=DEVICE))
    cnn_model.eval()

    fusion_model = FusionNet(feature_dim=metadata["feature_dim"], input_channels=signal_channels).to(DEVICE)
    fusion_model.load_state_dict(torch.load(artifact_path("fusion_model.pt"), map_location=DEVICE))
    fusion_model.eval()
    return scaler, xgb_model, cnn_model, fusion_model, meta_model, train_idx, test_idx, metadata
