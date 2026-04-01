import wfdb
import numpy as np
import os
from scipy.signal import find_peaks
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, signal, feature):
        # signal: (batch, 1, 3000)
        x = self.signal_conv(signal)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        x, _ = self.signal_lstm(x)
        x, _ = self.signal_lstm_2(x)
        x = x[:, -1, :]
        cnn_out = torch.relu(self.signal_fc(x))

        feat_out = self.feature_fc(feature)

        combined = torch.cat([cnn_out, feat_out], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits


def get_spectrogram(segment, fs=100):
    f, t, Sxx = spectrogram(segment, fs=fs)
    
    Sxx = np.log(Sxx + 1e-8)
    
    # 🔥 NORMALIZE EACH SPEC
    Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-8)
    
    
    return Sxx


def extract_features(segment, fs=100):
    features = []
    segment = np.nan_to_num(segment)
    
    # ---- Time domain ----
    features.append(np.mean(segment))
    features.append(np.std(segment))
    features.append(np.var(segment))
    features.append(np.sqrt(np.mean(segment**2)))  # RMS
    features.append(np.max(segment))
    features.append(np.min(segment))
    features.append(np.median(segment))
    features.append(np.percentile(segment, 25))
    features.append(np.percentile(segment, 75))
    
    # ---- Peak detection ----
    peaks, _ = find_peaks(segment, distance=fs*0.3)
    
    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / fs
        
        hr = len(peaks)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0
        
        nn50 = np.sum(np.abs(diff_rr) > 0.05)
        pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0

        # RR features
        max_rr = np.max(rr_intervals)
        min_rr = np.min(rr_intervals)
        median_rr = np.median(rr_intervals)
        q25_rr = np.percentile(rr_intervals, 25)
        q75_rr = np.percentile(rr_intervals, 75)

    else:
        hr, mean_rr, std_rr, rmssd, pnn50 = 0,0,0,0,0
        max_rr, min_rr, median_rr, q25_rr, q75_rr = 0,0,0,0,0

    features.extend([
        hr, mean_rr, std_rr, rmssd, pnn50,
        max_rr, min_rr, median_rr, q25_rr, q75_rr
    ])
    

    energy = np.sum(segment**2)
    features.append(energy)

    # entropy
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

    # ---- Frequency domain ----
    fft_vals = np.abs(np.fft.fft(segment))
    freqs = np.fft.fftfreq(len(segment), d=1/fs)

    # Only positive frequencies
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

    # Peak amplitude variability
    features.append(np.std(segment[peaks]) if len(peaks) > 2 else 0)

    # Signal range
    features.append(np.max(segment) - np.min(segment))

    # Absolute mean
    features.append(np.mean(np.abs(segment)))
    
    return features

data_dir = "apnea_data"
fs = 100
window_size = fs * 30
stride = fs * 10   # overlap

all_segments = []
all_labels = []

records = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]

for rec in records:
    try:
        annotation = wfdb.rdann(f"{data_dir}/{rec}", 'apn')
    except:
        continue  # skip unlabeled
    
    record = wfdb.rdrecord(f"{data_dir}/{rec}")
    
    signal = record.p_signal[:, 0]
    labels = annotation.symbol
    
    for start in range(0, len(signal) - window_size, stride):
        end = start + window_size
        
        segment = signal[start:end]
        
        if len(segment) == window_size:
            if np.std(segment) == 0:
                continue  # skip bad segment
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)  
            if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
                continue    
            if np.max(segment) - np.min(segment) < 0.05:
                continue
      
            all_segments.append(segment)
            center = (start + end) // 2
            label_idx = center // window_size
            if label_idx < len(labels):
                all_labels.append(1 if labels[label_idx] == 'A' else 0)

print("Total segments:", len(all_segments))
print("Total labels:", len(all_labels))

X_signal = np.array(all_segments)
y = np.array(all_labels)

print("Shape:", X_signal.shape, y.shape)

print("Apnea count:", np.sum(y))
print("Normal count:", len(y) - np.sum(y))

feature_data = []

for seg in X_signal:
    feature_data.append(extract_features(seg))

X_features = np.array(feature_data)

# remove NaNs if any slipped through
X_features = np.nan_to_num(X_features)

print("Feature shape:", X_features.shape)


scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

# SPLITING
indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

# Apply SAME split everywhere
X_train_f, X_test_f = X_features[train_idx], X_features[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)

scale = neg / pos
pos_weight = torch.tensor([scale], dtype=torch.float32).to(device)
scale_pos_weight=scale

model = XGBClassifier(
    n_estimators=1200,
    max_depth=12,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=scale,
    eval_metric='logloss'
)

model.fit(X_train_f, y_train)

preds = model.predict(X_test_f)  # ← IMPORTANT

xgb_train_prob = model.predict_proba(X_train_f)[:,1]
xgb_test_prob = model.predict_proba(X_test_f)[:,1]


print("XGBoost Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))



X_signal = X_signal[..., np.newaxis]

X_signal_train = X_signal[train_idx]
X_signal_test = X_signal[test_idx]

X_feat_train, X_feat_test = X_features[train_idx], X_features[test_idx]


# ------ CNN / LSTM (PyTorch)

X_signal_train_t = torch.tensor(X_signal_train.transpose(0, 2, 1), dtype=torch.float32)
X_signal_test_t = torch.tensor(X_signal_test.transpose(0, 2, 1), dtype=torch.float32)
X_feat_train_t = torch.tensor(X_feat_train, dtype=torch.float32)
X_feat_test_t = torch.tensor(X_feat_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_signal_train_t, X_feat_train_t, y_train_t)
test_dataset = TensorDataset(X_signal_test_t, X_feat_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

fusion_model = FusionNet(feature_dim=X_features.shape[1]).to(device)


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.0002)

epochs = 20
for epoch in range(epochs):
    fusion_model.train()
    running_loss = 0.0

    for sig_batch, feat_batch, y_batch in train_loader:
        sig_batch = sig_batch.to(device)
        feat_batch = feat_batch.to(device)
        y_batch = y_batch.to(device)

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
            sig_batch = sig_batch.to(device)
            feat_batch = feat_batch.to(device)
            y_batch = y_batch.to(device)

            logits = fusion_model(sig_batch, feat_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_acc = correct / total if total > 0 else 0.0
    print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")


def predict_probs(model, signal_arr, feature_arr, batch_size=64):
    model.eval()
    sig_t = torch.tensor(signal_arr.transpose(0, 2, 1), dtype=torch.float32)
    feat_t = torch.tensor(feature_arr, dtype=torch.float32)
    dummy_y = torch.zeros(len(sig_t), dtype=torch.float32)

    loader = DataLoader(TensorDataset(sig_t, feat_t, dummy_y), batch_size=batch_size, shuffle=False)
    all_probs = []

    with torch.no_grad():
        for sig_batch, feat_batch, _ in loader:
            sig_batch = sig_batch.to(device)
            feat_batch = feat_batch.to(device)
            logits = model(sig_batch, feat_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


cnn_train_prob = predict_probs(fusion_model, X_signal_train, X_feat_train)
cnn_test_prob = predict_probs(fusion_model, X_signal_test, X_feat_test)

fusion_preds = (cnn_test_prob >= 0.5).astype(int)
fusion_acc = accuracy_score(y_test, fusion_preds)
print("Fusion Accuracy:", fusion_acc)



meta_train = np.column_stack([xgb_train_prob, cnn_train_prob])
meta_test = np.column_stack([xgb_test_prob, cnn_test_prob])

# Split train into train + val for stacking
meta_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

meta_model.fit(meta_train, y_train)

final_pred = meta_model.predict(meta_test)
print("Stacked_accuracy:", accuracy_score(y_test, final_pred))




