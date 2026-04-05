import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import wfdb
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


DATA_DIR = "apnea_data"
ARTIFACT_DIR = "artifacts"
FS = 100
WINDOW_SECONDS = 30
STRIDE_SECONDS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 20
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
MC_DROPOUT_PASSES = 25

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
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
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
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


def ensure_artifact_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def artifact_path(name):
    return os.path.join(ARTIFACT_DIR, name)


def save_json(data, filename):
    ensure_artifact_dir()
    with open(artifact_path(filename), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(filename):
    with open(artifact_path(filename), "r", encoding="utf-8") as handle:
        return json.load(handle)


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

    fusion_model = FusionNet(feature_dim=feature_dim).to(DEVICE)
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


def train_cnn_baseline(x_signal_train, x_signal_test, y_train, y_test, pos_weight):
    x_signal_train_t = torch.tensor(x_signal_train.transpose(0, 2, 1), dtype=torch.float32)
    x_signal_test_t = torch.tensor(x_signal_test.transpose(0, 2, 1), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_signal_train_t, y_train_t)
    test_dataset = TensorDataset(x_signal_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = CNNBaseline().to(DEVICE)
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

    saliency = signal_tensor.grad.detach().abs().cpu().numpy()[0, 0]
    saliency = saliency / (saliency.max() + 1e-8)
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

    saliency = signal_tensor.grad.detach().abs().cpu().numpy()[0, 0]
    saliency = saliency / (saliency.max() + 1e-8)
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


def save_array(name, array):
    ensure_artifact_dir()
    np.save(artifact_path(name), array)


def load_array(name):
    return np.load(artifact_path(name), allow_pickle=False)


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
    cnn_model = CNNBaseline().to(DEVICE)
    cnn_model.load_state_dict(torch.load(artifact_path("cnn_model.pt"), map_location=DEVICE))
    cnn_model.eval()

    fusion_model = FusionNet(feature_dim=metadata["feature_dim"]).to(DEVICE)
    fusion_model.load_state_dict(torch.load(artifact_path("fusion_model.pt"), map_location=DEVICE))
    fusion_model.eval()
    return scaler, xgb_model, cnn_model, fusion_model, meta_model, train_idx, test_idx, metadata
