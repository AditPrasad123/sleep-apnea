import argparse

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pipeline import (
    DATA_DIR,
    FEATURE_NAMES,
    RANDOM_STATE,
    STRIDE_SECONDS,
    TEST_SIZE,
    WINDOW_SECONDS,
    FS,
    artifact_path,
    build_feature_matrix,
    build_train_test_split,
    compute_class_weights,
    ensure_artifact_dir,
    load_segments_and_labels,
    predict_probs,
    predict_probs_signal,
    save_array,
    save_json,
    train_cnn_baseline,
    train_fusion_model,
    train_stacking,
    train_xgboost,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sleep apnea models selectively.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "xgboost", "cnn", "fusionnet", "stacking"],
        default=["all"],
        help="Models to train. Use one or more: xgboost cnn fusionnet stacking. Default: all",
    )
    return parser.parse_args()


def resolve_requested_models(raw_models):
    if "all" in raw_models:
        return {"xgboost", "cnn", "fusionnet", "stacking"}
    return set(raw_models)


def save_common_artifacts(scaler, train_idx, test_idx, feature_dim):
    ensure_artifact_dir()
    joblib.dump(scaler, artifact_path("scaler.joblib"))
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


def main():
    args = parse_args()
    requested = resolve_requested_models(args.models)

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

    x_signal = x_signal[..., np.newaxis]
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


if __name__ == "__main__":
    main()
