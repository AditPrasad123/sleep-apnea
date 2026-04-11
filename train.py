import argparse
import os

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import (
    CROSS_ARTIFACT_DIR,
    DATA_DIR,
    FEATURE_NAMES,
    RANDOM_STATE,
    SIGNAL_CHANNELS,
    STRIDE_SECONDS,
    TEST_SIZE,
    WINDOW_SECONDS,
    FS,
    artifact_path,
    build_ecg_edr_signal,
    build_feature_matrix,
    build_train_test_split,
    compute_class_weights,
    ensure_artifact_dir,
    load_apnea_ecg_segments_30s,
    load_mitbih_psg_segments_30s,
    load_segments_and_labels,
    predict_probs,
    predict_probs_signal,
    save_array,
    save_json,
    summarize_split,
    tune_cross_threshold,
    train_cnn_baseline,
    train_fusion_model,
    train_stacking,
    train_xgboost,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sleep apnea models selectively.")
    parser.add_argument(
        "--mode",
        choices=["normal", "cross"],
        default="normal",
        help="Training mode. normal = Apnea-ECG train/test split, cross = train on Apnea-ECG and validate on MIT-BIH PSG.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "xgboost", "cnn", "fusionnet", "stacking"],
        default=["all"],
        help="Models to train. Use one or more: xgboost cnn fusionnet stacking. Default: all",
    )
    parser.add_argument("--apnea-dir", default=DATA_DIR, help="Path to Apnea-ECG directory.")
    parser.add_argument("--mit-dir", default="mitbih_psg_data", help="Path to MIT-BIH PSG directory.")
    parser.add_argument(
        "--mit-val-size",
        type=float,
        default=0.3,
        help="Validation split ratio within MIT-BIH PSG for cross-dataset threshold tuning.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "balanced_accuracy", "mcc"],
        default="f1",
        help="Metric used for threshold tuning during cross-dataset evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for cross-dataset MIT val/test split.",
    )
    parser.add_argument(
        "--cross-signal-mode",
        choices=["ecg", "ecg_edr"],
        default="ecg",
        help="Cross mode signal input for CNN/FusionNet/Stacking: ecg (1-channel) or ecg_edr (2-channel).",
    )
    parser.add_argument(
        "--cross-harmonize-level",
        choices=["none", "light", "full"],
        default="light",
        help="Cross preprocessing harmonization level. Default: light.",
    )
    parser.add_argument(
        "--no-cross-harmonize",
        action="store_true",
        help="Shortcut to set cross harmonization level to none.",
    )
    return parser.parse_args()


def cross_run_dir(signal_mode):
    return os.path.join(CROSS_ARTIFACT_DIR, signal_mode)


def resolve_requested_models(raw_models):
    if "all" in raw_models:
        return {"xgboost", "cnn", "fusionnet", "stacking"}
    return set(raw_models)


def save_common_artifacts(scaler, train_idx, test_idx, feature_dim, artifact_dir=None):
    base_dir = artifact_dir
    ensure_artifact_dir(base_dir=base_dir) if base_dir else ensure_artifact_dir()
    if base_dir:
        joblib.dump(scaler, artifact_path("scaler.joblib", base_dir=base_dir))
        save_array("train_idx.npy", train_idx, base_dir=base_dir)
        save_array("test_idx.npy", test_idx, base_dir=base_dir)
    else:
        joblib.dump(scaler, artifact_path("scaler.joblib"))
        save_array("train_idx.npy", train_idx)
        save_array("test_idx.npy", test_idx)
    payload = {
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
    }
    if base_dir:
        save_json(payload, "metadata.json", base_dir=base_dir)
    else:
        save_json(payload, "metadata.json")


def run_normal_training(requested):
    print("=== Stage 1: Loading and Segmenting ECG Data ===")
    x_signal, y = load_segments_and_labels(DATA_DIR)

    print("Total segments:", len(x_signal))
    print("Total labels:", len(y))
    print("Shape:", x_signal.shape, y.shape)
    print("Apnea count:", np.sum(y))
    print("Normal count:", len(y) - np.sum(y))

    print("=== Stage 2: Feature Extraction and Scaling ===")
    x_features = build_feature_matrix(x_signal)
    print("Feature shape:", x_features.shape)

    scaler = StandardScaler()
    x_features = scaler.fit_transform(x_features)

    print("=== Stage 3: Train/Test Split ===")
    train_idx, test_idx = build_train_test_split(y)

    x_train_f = x_features[train_idx]
    x_test_f = x_features[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    save_common_artifacts(scaler, train_idx, test_idx, feature_dim=x_features.shape[1])

    scale, pos_weight = compute_class_weights(y_train)

    xgb_train_prob = None
    xgb_test_prob = None
    cnn_train_prob = None
    cnn_test_prob = None

    x_signal = build_ecg_edr_signal(x_signal, fs=FS)
    x_signal_train = x_signal[train_idx]
    x_signal_test = x_signal[test_idx]
    x_feat_train = x_features[train_idx]
    x_feat_test = x_features[test_idx]

    if "xgboost" in requested or "stacking" in requested:
        print("=== Stage 4: Training Model 1 (XGBoost on Engineered Features) ===")
        xgb_model, xgb_train_prob, xgb_test_prob = train_xgboost(x_train_f, y_train, x_test_f, y_test, scale)
        joblib.dump(xgb_model, artifact_path("xgb_model.joblib"))
        print("Saved XGBoost artifacts.")

    if "cnn" in requested:
        print("=== Stage 5: Training Model 2 (CNN Baseline) ===")
        cnn_model = train_cnn_baseline(
            x_signal_train,
            x_signal_test,
            y_train,
            y_test,
            pos_weight,
            input_channels=x_signal.shape[2],
        )
        cnn_base_test_prob = predict_probs_signal(cnn_model, x_signal_test)
        cnn_base_preds = (cnn_base_test_prob >= 0.5).astype(int)
        cnn_base_acc = np.mean(cnn_base_preds == y_test)
        print("CNN Baseline Accuracy:", cnn_base_acc)
        torch.save(cnn_model.state_dict(), artifact_path("cnn_model.pt"))
        print("Saved CNN baseline artifacts.")

    if "fusionnet" in requested or "stacking" in requested:
        print("=== Stage 6: Training Model 3 (FusionNet: CNN + LSTM + Features) ===")
        fusion_model = train_fusion_model(
            x_signal_train,
            x_signal_test,
            x_feat_train,
            x_feat_test,
            y_train,
            y_test,
            pos_weight,
            feature_dim=x_features.shape[1],
            input_channels=x_signal.shape[2],
        )

        cnn_train_prob = predict_probs(fusion_model, x_signal_train, x_feat_train)
        cnn_test_prob = predict_probs(fusion_model, x_signal_test, x_feat_test)

        fusion_preds = (cnn_test_prob >= 0.5).astype(int)
        fusion_acc = np.mean(fusion_preds == y_test)
        print("Fusion Accuracy:", fusion_acc)
        torch.save(fusion_model.state_dict(), artifact_path("fusion_model.pt"))
        print("Saved FusionNet artifacts.")

    if "stacking" in requested:
        print("=== Stage 7: Training Model 4 (XGBoost + CNN + LSTM Stacking) ===")
        meta_model = train_stacking(
            xgb_train_prob,
            xgb_test_prob,
            cnn_train_prob,
            cnn_test_prob,
            y_train,
            y_test,
        )
        joblib.dump(meta_model, artifact_path("stacking_model.joblib"))
        print("Saved stacking artifacts.")

    print("Saved/updated model artifacts in artifacts/")


def run_cross_training(args, requested):
    print("=== Cross-Dataset Training: Loading Source and Target Data ===")
    harmonize_level = "none" if args.no_cross_harmonize else args.cross_harmonize_level
    x_train_sig, y_train = load_apnea_ecg_segments_30s(args.apnea_dir, fs=FS, harmonize_level=harmonize_level)
    x_target_sig, y_target = load_mitbih_psg_segments_30s(args.mit_dir, target_fs=FS, harmonize_level=harmonize_level)

    x_val_sig, x_test_sig, y_val, y_test = train_test_split(
        x_target_sig,
        y_target,
        test_size=1.0 - args.mit_val_size,
        random_state=args.random_state,
        stratify=y_target,
    )

    summarize_split("Apnea-ECG train", y_train)
    summarize_split("MIT-BIH PSG val", y_val)
    summarize_split("MIT-BIH PSG test", y_test)

    run_dir = cross_run_dir(args.cross_signal_mode)
    ensure_artifact_dir(base_dir=run_dir)

    if len(x_train_sig) == 0 or len(x_val_sig) == 0 or len(x_test_sig) == 0:
        raise RuntimeError("Empty split after preprocessing. Check paths and preprocessing filters.")

    x_train_feat_raw = build_feature_matrix(x_train_sig)
    x_val_feat_raw = build_feature_matrix(x_val_sig)

    scaler = StandardScaler()
    x_train_feat = scaler.fit_transform(x_train_feat_raw)
    x_val_feat = scaler.transform(x_val_feat_raw)

    joblib.dump(scaler, artifact_path("scaler.joblib", base_dir=run_dir))

    if args.cross_signal_mode == "ecg_edr":
        x_train_sig_n = build_ecg_edr_signal(x_train_sig, fs=FS)
        x_val_sig_n = build_ecg_edr_signal(x_val_sig, fs=FS)
        cross_signal_streams = ["ecg", "edr"]
    else:
        x_train_sig_n = x_train_sig[..., np.newaxis]
        x_val_sig_n = x_val_sig[..., np.newaxis]
        cross_signal_streams = ["ecg"]

    cross_signal_channels = int(x_train_sig_n.shape[2])

    scale, pos_weight = compute_class_weights(y_train)

    xgb_train_prob = None
    xgb_val_prob = None
    fusion_train_prob = None
    fusion_val_prob = None

    if "xgboost" in requested or "stacking" in requested:
        print("=== Cross Stage 1: Training XGBoost ===")
        xgb_model, xgb_train_prob, xgb_val_prob = train_xgboost(
            x_train_feat,
            y_train,
            x_val_feat,
            y_val,
            scale,
        )
        joblib.dump(xgb_model, artifact_path("xgb_model.joblib", base_dir=run_dir))

    if "cnn" in requested:
        print("=== Cross Stage 2: Training CNN ===")
        cnn_model = train_cnn_baseline(
            x_train_sig_n,
            x_val_sig_n,
            y_train,
            y_val,
            pos_weight,
            input_channels=cross_signal_channels,
        )
        torch.save(cnn_model.state_dict(), artifact_path("cnn_model.pt", base_dir=run_dir))

    if "fusionnet" in requested or "stacking" in requested:
        print("=== Cross Stage 3: Training FusionNet ===")
        fusion_model = train_fusion_model(
            x_train_sig_n,
            x_val_sig_n,
            x_train_feat,
            x_val_feat,
            y_train,
            y_val,
            pos_weight,
            feature_dim=x_train_feat.shape[1],
            input_channels=cross_signal_channels,
        )
        fusion_train_prob = predict_probs(fusion_model, x_train_sig_n, x_train_feat)
        fusion_val_prob = predict_probs(fusion_model, x_val_sig_n, x_val_feat)
        torch.save(fusion_model.state_dict(), artifact_path("fusion_model.pt", base_dir=run_dir))

    if "stacking" in requested:
        print("=== Cross Stage 4: Training Stacking Meta Model ===")
        meta_model = train_stacking(
            xgb_train_prob,
            xgb_val_prob,
            fusion_train_prob,
            fusion_val_prob,
            y_train,
            y_val,
        )
        joblib.dump(meta_model, artifact_path("stacking_model.joblib", base_dir=run_dir))

    threshold_info = {}
    if xgb_val_prob is not None:
        threshold, score = tune_cross_threshold(y_val, xgb_val_prob, args.threshold_metric)
        threshold_info["xgboost"] = {
            "metric": args.threshold_metric,
            "threshold": threshold,
            "val_score": score,
        }
    if fusion_val_prob is not None:
        threshold, score = tune_cross_threshold(y_val, fusion_val_prob, args.threshold_metric)
        threshold_info["fusionnet"] = {
            "metric": args.threshold_metric,
            "threshold": threshold,
            "val_score": score,
        }

    save_json(
        {
            "mode": "cross",
            "artifact_dir": run_dir,
            "fs": FS,
            "apnea_dir": args.apnea_dir,
            "mit_dir": args.mit_dir,
            "mit_val_size": args.mit_val_size,
            "random_state": args.random_state,
            "threshold_metric": args.threshold_metric,
            "harmonize_preprocessing": harmonize_level != "none",
            "cross_harmonize_level": harmonize_level,
            "cross_signal_mode": args.cross_signal_mode,
            "feature_dim": int(x_train_feat.shape[1]),
            "feature_names": FEATURE_NAMES,
            "signal_channels": cross_signal_channels,
            "signal_streams": cross_signal_streams,
            "source_train_size": int(len(y_train)),
            "target_val_size": int(len(y_val)),
            "target_test_size": int(len(y_test)),
            "trained_models": sorted(list(requested)),
            "precomputed_thresholds": threshold_info,
        },
        "metadata.json",
        base_dir=run_dir,
    )
    print(f"Saved/updated cross-dataset model artifacts in {run_dir}/")


def main():
    args = parse_args()
    requested = resolve_requested_models(args.models)

    if args.mode == "normal":
        run_normal_training(requested)
    else:
        run_cross_training(args, requested)


if __name__ == "__main__":
    main()
