import wfdb
import numpy as np
import os
from scipy.signal import find_peaks
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from scipy.stats import skew, kurtosis



def transformer_block(x, head_size=64, num_heads=4, ff_dim=128):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size
    )(x, x)
    
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(x.shape[-1])(ff)
    
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    
    return x


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
    
    
    # RR features
    features.append(np.max(rr_intervals) if len(peaks)>2 else 0)
    features.append(np.min(rr_intervals) if len(peaks)>2 else 0)

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
            all_segments.append(segment)
            label_idx = start // window_size
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


model = XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(23514 / 14664),
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


# ------ CNN / LTSM

signal_input = layers.Input(shape=(3000, 1))

x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(signal_input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)

# LSTM for temporal patterns
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(32))(x)

cnn_out = layers.Dense(64, activation='relu')(x)

# ---- FEATURE BRANCH ----
feature_input = layers.Input(shape=(X_features.shape[1],))

f = layers.Dense(64, activation='relu')(feature_input)
f = layers.Dense(32, activation='relu')(f)

# ---- FUSION ----
combined = layers.concatenate([cnn_out, f])

z = layers.Dense(64, activation='relu')(combined)
z = layers.Dense(1, activation='sigmoid')(z)

model = models.Model(inputs=[signal_input, feature_input], outputs=z)

model.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
class_weight = {
    0: 1.0,
    1: 23514 / 14664
}

model.fit(
    [X_signal_train, X_feat_train],
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=([X_signal_test, X_feat_test], y_test),
    class_weight=class_weight
)

cnn_train_prob = model.predict([X_signal_train, X_feat_train]).flatten()
cnn_test_prob = model.predict([X_signal_test, X_feat_test]).flatten()

loss, acc = model.evaluate([X_signal_test, X_feat_test], y_test)
print("Fusion Accuracy:", acc)



meta_train = np.column_stack([xgb_train_prob, cnn_train_prob])
meta_test = np.column_stack([xgb_test_prob, cnn_test_prob])

# Split train into train + val for stacking
meta_model = LogisticRegression(C=0.5)
meta_model.fit(meta_train, y_train)

final_pred = meta_model.predict(meta_test)
print("Stacked_accuracy:", accuracy_score(y_test, final_pred))




