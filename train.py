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
    build_signal_view,
    build_ecg_edr_signal,
    build_feature_matrix,
    build_train_test_split,
    compare_threshold_metrics,
    compute_class_weights,
    ensure_artifact_dir,
    fine_tune_fusion_model,
    fine_tune_signal_model,
    fine_tune_xgboost_model,
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
    train_chunk_cnn_lstm,
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
        choices=["all", "xgboost", "cnn", "chunknet", "fusionnet", "stacking"],
        default=["all"],
        help="Models to train. Use one or more: xgboost cnn chunknet fusionnet stacking. Default: all",
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
        choices=["ecg", "edr", "ecg_edr"],
        default="ecg",
        help="Cross mode signal input for CNN/FusionNet/Stacking: ecg, edr, or ecg_edr.",
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
    parser.add_argument(
        "--few-shot-mit-frac",
        type=float,
        default=0.0,
        help="Fraction of MIT validation split to use for few-shot target adaptation (0 disables).",
    )
    parser.add_argument(
        "--few-shot-epochs",
        type=int,
        default=5,
        help="Few-shot fine-tuning epochs for neural models.",
    )
    parser.add_argument(
        "--few-shot-lr",
        type=float,
        default=1e-4,
        help="Few-shot fine-tuning learning rate for neural models.",
    )
    return parser.parse_args()


def cross_run_dir(signal_mode):
    return os.path.join(CROSS_ARTIFACT_DIR, signal_mode)


def cross_artifact_root(args):
    if args.few_shot_mit_frac <= 0.0:
        return CROSS_ARTIFACT_DIR

    few_shot_tag = f"frac_{args.few_shot_mit_frac:g}_ep_{args.few_shot_epochs}_lr_{args.few_shot_lr:g}"
    return os.path.join(CROSS_ARTIFACT_DIR, "few_shot", few_shot_tag)


def harmonize_dir_name(harmonize_level):
    mapping = {
        "none": "no_harmonization",
        "light": "light_harmonization",
        "full": "full_harmonization",
    }
    return mapping[harmonize_level]


def cross_model_run_dir(artifact_root, model_key, signal_mode, harmonize_level):
    return os.path.join(artifact_root, model_key, signal_mode, harmonize_dir_name(harmonize_level))


def resolve_requested_models(raw_models):
    if "all" in raw_models:
        return {"xgboost", "cnn", "chunknet", "fusionnet", "stacking"}
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

    if "chunknet" in requested:
        print("=== Stage 5b: Training Model 2b (ChunkCNNLSTM: 5-second CNN chunks + LSTM) ===")
        chunk_model = train_chunk_cnn_lstm(
            x_signal_train,
            x_signal_test,
            y_train,
            y_test,
            pos_weight,
            input_channels=x_signal.shape[2],
        )
        chunk_test_prob = predict_probs_signal(chunk_model, x_signal_test)
        chunk_preds = (chunk_test_prob >= 0.5).astype(int)
        chunk_acc = np.mean(chunk_preds == y_test)
        print("ChunkCNNLSTM Accuracy:", chunk_acc)
        torch.save(chunk_model.state_dict(), artifact_path("chunk_model.pt"))
        print("Saved ChunkCNNLSTM artifacts.")

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

    use_few_shot = args.few_shot_mit_frac > 0.0
    if use_few_shot:
        if not (0.0 < args.few_shot_mit_frac < 1.0):
            raise ValueError("--few-shot-mit-frac must be in (0, 1).")

        x_adapt_sig, x_val_sig, y_adapt, y_val = train_test_split(
            x_val_sig,
            y_val,
            test_size=1.0 - args.few_shot_mit_frac,
            random_state=args.random_state,
            stratify=y_val,
        )
    else:
        x_adapt_sig = np.empty((0, x_val_sig.shape[1]), dtype=x_val_sig.dtype)
        y_adapt = np.empty((0,), dtype=y_val.dtype)

    summarize_split("Apnea-ECG train", y_train)
    if use_few_shot:
        summarize_split("MIT-BIH PSG few-shot adapt", y_adapt)
    summarize_split("MIT-BIH PSG val", y_val)
    summarize_split("MIT-BIH PSG test", y_test)

    artifact_root = cross_artifact_root(args)
    model_dirs = {
        model_key: cross_model_run_dir(artifact_root, model_key, args.cross_signal_mode, harmonize_level)
        for model_key in ["xgboost", "cnn", "chunknet", "fusionnet", "stacking"]
    }
    for model_dir in model_dirs.values():
        ensure_artifact_dir(base_dir=model_dir)

    if len(x_train_sig) == 0 or len(x_val_sig) == 0 or len(x_test_sig) == 0:
        raise RuntimeError("Empty split after preprocessing. Check paths and preprocessing filters.")

    x_train_feat_raw = build_feature_matrix(x_train_sig)
    x_adapt_feat_raw = build_feature_matrix(x_adapt_sig) if use_few_shot else np.empty((0, len(FEATURE_NAMES)))
    x_val_feat_raw = build_feature_matrix(x_val_sig)

    scaler = StandardScaler()
    x_train_feat = scaler.fit_transform(x_train_feat_raw)
    x_adapt_feat = scaler.transform(x_adapt_feat_raw) if use_few_shot else np.empty((0, x_train_feat.shape[1]))
    x_val_feat = scaler.transform(x_val_feat_raw)

    for model_dir in model_dirs.values():
        joblib.dump(scaler, artifact_path("scaler.joblib", base_dir=model_dir))

    x_train_sig_n, cross_signal_streams = build_signal_view(
        x_train_sig,
        args.cross_signal_mode,
        fs=FS,
    )
    x_val_sig_n, _ = build_signal_view(
        x_val_sig,
        args.cross_signal_mode,
        fs=FS,
    )
    if use_few_shot:
        x_adapt_sig_n, _ = build_signal_view(
            x_adapt_sig,
            args.cross_signal_mode,
            fs=FS,
        )
    else:
        x_adapt_sig_n = np.empty((0, x_train_sig_n.shape[1], x_train_sig_n.shape[2]), dtype=np.float32)

    cross_signal_channels = int(x_train_sig_n.shape[2])

    scale, pos_weight = compute_class_weights(y_train)

    xgb_train_prob = None
    xgb_val_prob = None
    cnn_val_prob = None
    chunk_val_prob = None
    fusion_train_prob = None
    fusion_val_prob = None
    meta_val_prob = None

    if "xgboost" in requested or "stacking" in requested:
        print("=== Cross Stage 1: Training XGBoost ===")
        xgb_model, xgb_train_prob, xgb_val_prob = train_xgboost(
            x_train_feat,
            y_train,
            x_val_feat,
            y_val,
            scale,
        )
        if use_few_shot:
            print("=== Cross Stage 1b: Few-shot MIT fine-tuning (XGBoost) ===")
            xgb_model = fine_tune_xgboost_model(xgb_model, x_adapt_feat, y_adapt)
            xgb_train_prob = xgb_model.predict_proba(x_train_feat)[:, 1]
            xgb_val_prob = xgb_model.predict_proba(x_val_feat)[:, 1]
        joblib.dump(xgb_model, artifact_path("xgb_model.joblib", base_dir=model_dirs["xgboost"]))

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
        if use_few_shot:
            print("=== Cross Stage 2a: Few-shot MIT fine-tuning (CNN) ===")
            cnn_model = fine_tune_signal_model(
                cnn_model,
                x_adapt_sig_n,
                y_adapt,
                pos_weight,
                epochs=args.few_shot_epochs,
                learning_rate=args.few_shot_lr,
            )
        cnn_val_prob = predict_probs_signal(cnn_model, x_val_sig_n)
        torch.save(cnn_model.state_dict(), artifact_path("cnn_model.pt", base_dir=model_dirs["cnn"]))

    if "chunknet" in requested:
        print("=== Cross Stage 2b: Training ChunkCNNLSTM ===")
        chunk_model = train_chunk_cnn_lstm(
            x_train_sig_n,
            x_val_sig_n,
            y_train,
            y_val,
            pos_weight,
            input_channels=cross_signal_channels,
        )
        if use_few_shot:
            print("=== Cross Stage 2c: Few-shot MIT fine-tuning (ChunkCNNLSTM) ===")
            chunk_model = fine_tune_signal_model(
                chunk_model,
                x_adapt_sig_n,
                y_adapt,
                pos_weight,
                epochs=args.few_shot_epochs,
                learning_rate=args.few_shot_lr,
            )
        chunk_val_prob = predict_probs_signal(chunk_model, x_val_sig_n)
        torch.save(chunk_model.state_dict(), artifact_path("chunk_model.pt", base_dir=model_dirs["chunknet"]))

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
        if use_few_shot:
            print("=== Cross Stage 3b: Few-shot MIT fine-tuning (FusionNet) ===")
            fusion_model = fine_tune_fusion_model(
                fusion_model,
                x_adapt_sig_n,
                x_adapt_feat,
                y_adapt,
                pos_weight,
                epochs=args.few_shot_epochs,
                learning_rate=args.few_shot_lr,
            )
            fusion_train_prob = predict_probs(fusion_model, x_train_sig_n, x_train_feat)
            fusion_val_prob = predict_probs(fusion_model, x_val_sig_n, x_val_feat)
        torch.save(fusion_model.state_dict(), artifact_path("fusion_model.pt", base_dir=model_dirs["fusionnet"]))

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
        meta_val_prob = meta_model.predict_proba(np.column_stack([xgb_val_prob, fusion_val_prob]))[:, 1]
        joblib.dump(meta_model, artifact_path("stacking_model.joblib", base_dir=model_dirs["stacking"]))

    threshold_info = {}
    threshold_comparison = {}
    if xgb_val_prob is not None:
        threshold, score = tune_cross_threshold(y_val, xgb_val_prob, args.threshold_metric)
        threshold_info["xgboost"] = {
            "metric": args.threshold_metric,
            "threshold": threshold,
            "val_score": score,
        }
        threshold_comparison["xgboost"] = compare_threshold_metrics(y_val, xgb_val_prob)
    if cnn_val_prob is not None:
        threshold, score = tune_cross_threshold(y_val, cnn_val_prob, args.threshold_metric)
        threshold_info["cnn"] = {
            "metric": args.threshold_metric,
            "threshold": threshold,
            "val_score": score,
        }
        threshold_comparison["cnn"] = compare_threshold_metrics(y_val, cnn_val_prob)
    if chunk_val_prob is not None:
        threshold, score = tune_cross_threshold(y_val, chunk_val_prob, args.threshold_metric)
        threshold_info["chunknet"] = {
            "metric": args.threshold_metric,
            "threshold": threshold,
            "val_score": score,
        }
        threshold_comparison["chunknet"] = compare_threshold_metrics(y_val, chunk_val_prob)
    if fusion_val_prob is not None:
        threshold, score = tune_cross_threshold(y_val, fusion_val_prob, args.threshold_metric)
        threshold_info["fusionnet"] = {
            "metric": args.threshold_metric,
            "threshold": threshold,
            "val_score": score,
        }
        threshold_comparison["fusionnet"] = compare_threshold_metrics(y_val, fusion_val_prob)
    if meta_val_prob is not None:
        threshold, score = tune_cross_threshold(y_val, meta_val_prob, args.threshold_metric)
        threshold_info["stacking"] = {
            "metric": args.threshold_metric,
            "threshold": threshold,
            "val_score": score,
        }
        threshold_comparison["stacking"] = compare_threshold_metrics(y_val, meta_val_prob)

    for model_key, model_dir in model_dirs.items():
        model_threshold_info = threshold_info.get(model_key)
        model_threshold_comparison = threshold_comparison.get(model_key)

        save_json(
            {
                "target_dataset": "mitbih_psg",
                "selection_metric": args.threshold_metric,
                "signal_mode": args.cross_signal_mode,
                "harmonize_level": harmonize_level,
                "model": model_key,
                "thresholds": model_threshold_comparison,
            },
            "threshold_calibration.json",
            base_dir=model_dir,
        )

        save_json(
            {
                "mode": "cross",
                "artifact_dir": model_dir,
                "artifact_root": artifact_root,
                "model": model_key,
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
                "few_shot_mit_frac": float(args.few_shot_mit_frac),
                "few_shot_adapt_size": int(len(y_adapt)),
                "few_shot_epochs": int(args.few_shot_epochs),
                "few_shot_lr": float(args.few_shot_lr),
                "few_shot_enabled": bool(use_few_shot),
                "trained_models": sorted(list(requested)),
                "precomputed_threshold": model_threshold_info,
                "threshold_metric_comparison": model_threshold_comparison,
            },
            "metadata.json",
            base_dir=model_dir,
        )

    print(f"Saved/updated cross-dataset model artifacts in {artifact_root}/")


def main():
    args = parse_args()
    requested = resolve_requested_models(args.models)

    if args.mode == "normal":
        run_normal_training(requested)
    else:
        run_cross_training(args, requested)


if __name__ == "__main__":
    main()
