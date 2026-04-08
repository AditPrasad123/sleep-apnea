import argparse
import os

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline import (
    FS,
    build_feature_matrix,
    compute_class_weights,
    predict_probs,
    predict_probs_signal,
    train_cnn_baseline,
    train_fusion_model,
)


APNEA_TOKENS = {"OA", "X", "CA", "CAA", "H", "HA"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-dataset ECG-only baseline: train on Apnea-ECG, tune threshold on MIT-BIH val, test on MIT-BIH test"
    )
    parser.add_argument("--apnea-dir", default="apnea_data", help="Path to Apnea-ECG data directory")
    parser.add_argument("--mit-dir", default="mitbih_psg_data", help="Path to MIT-BIH PSG data directory")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "xgboost", "cnn", "fusionnet", "stacking"],
        default=["xgboost"],
        help="Model(s) to run. Default: xgboost",
    )
    parser.add_argument(
        "--mit-val-size",
        type=float,
        default=0.3,
        help="Validation split ratio within MIT-BIH for threshold tuning (default: 0.3).",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "balanced_accuracy", "mcc"],
        default="f1",
        help="Metric used to select threshold on MIT-BIH validation set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for MIT-BIH val/test split.",
    )
    return parser.parse_args()


def resolve_models(raw_models):
    if "all" in raw_models:
        return {"xgboost", "cnn", "fusionnet", "stacking"}
    return set(raw_models)


def normalize_segment(segment):
    segment = np.nan_to_num(segment.astype(np.float32))
    std = np.std(segment)
    if std <= 0:
        return None
    segment = (segment - np.mean(segment)) / (std + 1e-8)
    if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
        return None
    if np.max(segment) - np.min(segment) < 0.05:
        return None
    return segment


def load_apnea_ecg_segments_30s(data_dir, fs=FS):
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

            segment = normalize_segment(signal[start:end])
            if segment is None:
                continue

            label = 1 if minute_labels[minute_idx] == "A" else 0
            segments.append(segment)
            labels.append(label)

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)


def tokenize_aux(note):
    return [token.strip().upper() for token in note.strip().split() if token.strip()]


def has_apnea_event(note):
    tokens = tokenize_aux(note)
    return any(token in APNEA_TOKENS for token in tokens)


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


def load_mitbih_psg_segments_30s(data_dir, target_fs=FS):
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
            segment = normalize_segment(resampled)
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


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-8)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": auc,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "specificity": specificity,
    }


def tune_threshold(y_true, y_prob, metric_name):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -np.inf

    metric_key = {
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "mcc": "mcc",
    }[metric_name]

    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold)
        score = metrics[metric_key]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def report_metrics(name, y_val, y_val_prob, y_test, y_test_prob, threshold_metric):
    tuned_threshold, best_val_score = tune_threshold(y_val, y_val_prob, threshold_metric)
    test_metrics = compute_metrics(y_test, y_test_prob, tuned_threshold)

    print(f"\n{name} Cross-Dataset Metrics")
    print(f"  tuned_threshold ({threshold_metric}): {tuned_threshold:.2f}")
    print(f"  val_{threshold_metric}: {best_val_score:.4f}")
    print("  test_accuracy:", test_metrics["accuracy"])
    print("  test_precision:", test_metrics["precision"])
    print("  test_recall:", test_metrics["recall"])
    print("  test_f1-score:", test_metrics["f1"])
    print("  test_auc-roc:", test_metrics["auc_roc"])
    print("  test_balanced_accuracy:", test_metrics["balanced_accuracy"])
    print("  test_mcc:", test_metrics["mcc"])
    print("  test_specificity:", test_metrics["specificity"])


def train_xgb_on_features(x_train_feat, y_train):
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=1200,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    model.fit(x_train_feat, y_train)
    return model


def run_xgboost(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, threshold_metric):
    x_train_feat = build_feature_matrix(x_train_sig)
    x_val_feat = build_feature_matrix(x_val_sig)
    x_test_feat = build_feature_matrix(x_test_sig)

    scaler = StandardScaler()
    x_train_feat = scaler.fit_transform(x_train_feat)
    x_val_feat = scaler.transform(x_val_feat)
    x_test_feat = scaler.transform(x_test_feat)

    model = train_xgb_on_features(x_train_feat, y_train)
    y_val_prob = model.predict_proba(x_val_feat)[:, 1]
    y_test_prob = model.predict_proba(x_test_feat)[:, 1]
    report_metrics("XGBoost", y_val, y_val_prob, y_test, y_test_prob, threshold_metric)


def run_cnn(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, threshold_metric):
    x_train = x_train_sig[..., np.newaxis]
    x_val = x_val_sig[..., np.newaxis]
    x_test = x_test_sig[..., np.newaxis]

    _, pos_weight = compute_class_weights(y_train)
    model = train_cnn_baseline(
        x_train,
        x_val,
        y_train,
        y_val,
        pos_weight,
        input_channels=1,
    )
    y_val_prob = predict_probs_signal(model, x_val)
    y_test_prob = predict_probs_signal(model, x_test)
    report_metrics("CNN", y_val, y_val_prob, y_test, y_test_prob, threshold_metric)


def run_fusionnet(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, threshold_metric):
    x_train_feat = build_feature_matrix(x_train_sig)
    x_val_feat = build_feature_matrix(x_val_sig)
    x_test_feat = build_feature_matrix(x_test_sig)

    scaler = StandardScaler()
    x_train_feat = scaler.fit_transform(x_train_feat)
    x_val_feat = scaler.transform(x_val_feat)
    x_test_feat = scaler.transform(x_test_feat)

    x_train = x_train_sig[..., np.newaxis]
    x_val = x_val_sig[..., np.newaxis]
    x_test = x_test_sig[..., np.newaxis]

    _, pos_weight = compute_class_weights(y_train)
    model = train_fusion_model(
        x_train,
        x_val,
        x_train_feat,
        x_val_feat,
        y_train,
        y_val,
        pos_weight,
        feature_dim=x_train_feat.shape[1],
        input_channels=1,
    )

    y_val_prob = predict_probs(model, x_val, x_val_feat)
    y_test_prob = predict_probs(model, x_test, x_test_feat)
    report_metrics("FusionNet", y_val, y_val_prob, y_test, y_test_prob, threshold_metric)


def run_stacking(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, threshold_metric):
    x_train_feat = build_feature_matrix(x_train_sig)
    x_val_feat = build_feature_matrix(x_val_sig)
    x_test_feat = build_feature_matrix(x_test_sig)

    scaler = StandardScaler()
    x_train_feat = scaler.fit_transform(x_train_feat)
    x_val_feat = scaler.transform(x_val_feat)
    x_test_feat = scaler.transform(x_test_feat)

    xgb_model = train_xgb_on_features(x_train_feat, y_train)
    xgb_train_prob = xgb_model.predict_proba(x_train_feat)[:, 1]
    xgb_val_prob = xgb_model.predict_proba(x_val_feat)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(x_test_feat)[:, 1]

    x_train = x_train_sig[..., np.newaxis]
    x_val = x_val_sig[..., np.newaxis]
    x_test = x_test_sig[..., np.newaxis]

    _, pos_weight = compute_class_weights(y_train)
    fusion_model = train_fusion_model(
        x_train,
        x_val,
        x_train_feat,
        x_val_feat,
        y_train,
        y_val,
        pos_weight,
        feature_dim=x_train_feat.shape[1],
        input_channels=1,
    )

    fusion_train_prob = predict_probs(fusion_model, x_train, x_train_feat)
    fusion_val_prob = predict_probs(fusion_model, x_val, x_val_feat)
    fusion_test_prob = predict_probs(fusion_model, x_test, x_test_feat)

    meta_train = np.column_stack([xgb_train_prob, fusion_train_prob])
    meta_val = np.column_stack([xgb_val_prob, fusion_val_prob])
    meta_test = np.column_stack([xgb_test_prob, fusion_test_prob])

    meta_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    meta_model.fit(meta_train, y_train)

    y_val_prob = meta_model.predict_proba(meta_val)[:, 1]
    y_test_prob = meta_model.predict_proba(meta_test)[:, 1]
    report_metrics("Stacking", y_val, y_val_prob, y_test, y_test_prob, threshold_metric)


def main():
    args = parse_args()
    models = resolve_models(args.models)

    print("Loading Apnea-ECG training data (30s ECG-only)...")
    x_train_sig, y_train = load_apnea_ecg_segments_30s(args.apnea_dir)
    summarize_split("Apnea-ECG train", y_train)

    print("Loading MIT-BIH PSG target-domain data (30s ECG-only)...")
    x_target_sig, y_target = load_mitbih_psg_segments_30s(args.mit_dir)

    x_val_sig, x_test_sig, y_val, y_test = train_test_split(
        x_target_sig,
        y_target,
        test_size=1.0 - args.mit_val_size,
        random_state=args.random_state,
        stratify=y_target,
    )

    summarize_split("MIT-BIH PSG val", y_val)
    summarize_split("MIT-BIH PSG test", y_test)

    if len(x_train_sig) == 0 or len(x_val_sig) == 0 or len(x_test_sig) == 0:
        raise RuntimeError("Empty split after preprocessing. Check paths and preprocessing filters.")

    if "xgboost" in models:
        run_xgboost(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, args.threshold_metric)

    if "cnn" in models:
        run_cnn(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, args.threshold_metric)

    if "fusionnet" in models:
        run_fusionnet(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, args.threshold_metric)

    if "stacking" in models:
        run_stacking(x_train_sig, y_train, x_val_sig, y_val, x_test_sig, y_test, args.threshold_metric)


if __name__ == "__main__":
    main()
