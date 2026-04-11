import argparse
import json
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from pipeline import (
    ChunkCNNLSTM,
    CNNBaseline,
    CROSS_ARTIFACT_DIR,
    DATA_DIR,
    DEVICE,
    FS,
    FusionNet,
    artifact_path,
    build_signal_view,
    build_ecg_edr_signal,
    build_feature_matrix,
    compare_threshold_metrics,
    compute_cross_metrics,
    compute_signal_saliency,
    compute_signal_saliency_signal_only,
    load_apnea_ecg_segments_30s,
    load_array,
    load_json,
    load_mitbih_psg_segments_30s,
    load_segments_and_labels,
    predict_probs_mc_dropout,
    predict_probs,
    predict_probs_signal,
    predict_probs_signal_mc_dropout,
    summarize_split,
    tune_cross_threshold,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sleep apnea models selectively.")
    parser.add_argument(
        "--mode",
        choices=["normal", "cross"],
        default="normal",
        help="Evaluation mode. normal = local held-out test set, cross = MIT-BIH val/test evaluation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "xgboost", "cnn", "chunknet", "fusionnet", "stacking"],
        default=["all"],
        help="Models to evaluate. Use one or more: xgboost cnn chunknet fusionnet stacking. Default: all",
    )
    parser.add_argument("--apnea-dir", default=DATA_DIR, help="Path to Apnea-ECG directory (cross mode).")
    parser.add_argument("--mit-dir", default="mitbih_psg_data", help="Path to MIT-BIH PSG directory (cross mode).")
    parser.add_argument(
        "--mit-val-size",
        type=float,
        default=0.3,
        help="Validation split ratio within MIT-BIH PSG for threshold tuning (cross mode).",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "balanced_accuracy", "mcc"],
        default="f1",
        help="Metric used to tune threshold on MIT-BIH val set (cross mode).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for cross-dataset MIT val/test split.",
    )
    parser.add_argument(
        "--cross-signal-mode",
        choices=["ecg", "edr", "ecg_edr"],
        default="ecg",
        help="Cross mode signal input (ecg, edr, or ecg_edr). Default: ecg.",
    )
    parser.add_argument(
        "--cross-harmonize-level",
        choices=["none", "light", "full"],
        default=None,
        help="Optional harmonization level override for cross evaluation. Default uses run metadata.",
    )
    parser.add_argument(
        "--no-cross-harmonize",
        action="store_true",
        help="Shortcut to set cross harmonization level to none.",
    )
    return parser.parse_args()


def cross_run_dir(signal_mode):
    return os.path.join(CROSS_ARTIFACT_DIR, signal_mode)


def harmonize_dir_name(harmonize_level):
    mapping = {
        "none": "no_harmonization",
        "light": "light_harmonization",
        "full": "full_harmonization",
    }
    return mapping[harmonize_level]


def cross_model_run_dir(model_key, signal_mode, harmonize_level):
    return os.path.join(CROSS_ARTIFACT_DIR, model_key, signal_mode, harmonize_dir_name(harmonize_level))


def resolve_requested_models(raw_models):
    if "all" in raw_models:
        return {"xgboost", "cnn", "chunknet", "fusionnet", "stacking"}
    return set(raw_models)


def ensure_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact missing: {path}")


def load_common_artifacts():
    scaler_path = artifact_path("scaler.joblib")
    metadata_path = artifact_path("metadata.json")
    test_idx_path = artifact_path("test_idx.npy")

    ensure_file_exists(scaler_path)
    ensure_file_exists(metadata_path)
    ensure_file_exists(test_idx_path)

    scaler = joblib.load(scaler_path)
    metadata = load_json("metadata.json")
    test_idx = load_array("test_idx.npy")
    return scaler, metadata, test_idx


def report_metrics(y_true, y_prob, y_pred, label):
    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"{label} Metrics:")
    print("  accuracy:", accuracy_score(y_true, y_pred))
    print("  precision:", precision_score(y_true, y_pred, zero_division=0))
    print("  recall:", recall_score(y_true, y_pred, zero_division=0))
    print("  f1-score:", f1_score(y_true, y_pred, zero_division=0))
    print("  auc-roc:", auc_roc)
    print("  pr-auc:", pr_auc)


def build_curves(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
    }


def ensure_model_dir(model_label, base_dir="artifacts"):
    model_dir = os.path.join(base_dir, model_label.lower().replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_roc_curve(model_dir, model_label, curves):
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(curves["fpr"], curves["tpr"], linewidth=2, label=f"AUC = {curves['auc_roc']:.3f}")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set_title(f"ROC Curve - {model_label}")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "roc_curve.png"), dpi=200)
    plt.close(fig)


def save_pr_curve(model_dir, model_label, curves):
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(curves["recall"], curves["precision"], linewidth=2, label=f"AP = {curves['pr_auc']:.3f}")
    axis.set_title(f"Precision-Recall Curve - {model_label}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "pr_curve.png"), dpi=200)
    plt.close(fig)


def save_confusion_matrix(model_dir, model_label, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set_title(f"Confusion Matrix - {model_label}")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Normal", "Apnea"])
    axis.set_yticklabels(["Normal", "Apnea"])

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                format(matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)


def save_model_outputs(model_label, y_true, y_prob, y_pred, base_dir="artifacts", model_dir=None):
    if model_dir is None:
        model_dir = ensure_model_dir(model_label, base_dir=base_dir)
    else:
        os.makedirs(model_dir, exist_ok=True)
    curves = build_curves(y_true, y_prob)
    save_roc_curve(model_dir, model_label, curves)
    save_pr_curve(model_dir, model_label, curves)
    save_confusion_matrix(model_dir, model_label, y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(curves["auc_roc"]),
        "pr_auc": float(curves["pr_auc"]),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return curves


def save_feature_importance(model_dir, feature_names, importances):
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[index] for index in order]
    sorted_importances = importances[order]

    top_k = min(15, len(sorted_importances))
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        sorted_names[:top_k][::-1],
        sorted_importances[:top_k][::-1],
        color="#2a6f97",
    )
    axis.set_title("XGBoost Feature Importance")
    axis.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "feature_importance.png"), dpi=200)
    plt.close(fig)

    feature_payload = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(sorted_names, sorted_importances)
    ]
    with open(os.path.join(model_dir, "feature_importance.json"), "w", encoding="utf-8") as handle:
        json.dump(feature_payload, handle, indent=2)


def save_shap_outputs(model_dir, model, x_data, feature_names, prefix="xgboost", max_samples=400):
    try:
        import shap
    except Exception as error:
        print(f"Skipping SHAP for {prefix}: {error}")
        return

    sample_count = min(max_samples, len(x_data))
    if sample_count == 0:
        return
    x_sample = x_data[:sample_count]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
    except Exception as error:
        print(f"Skipping SHAP computation for {prefix}: {error}")
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    np.save(os.path.join(model_dir, f"{prefix}_shap_values.npy"), np.asarray(shap_values))

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        x_sample,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{prefix}_shap_summary.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        x_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{prefix}_shap_bar.png"), dpi=200)
    plt.close()


def save_uncertainty_outputs(model_dir, mean_probs, std_probs):
    confidence = 1.0 - np.abs(mean_probs - 0.5) * 2.0
    confidence = np.clip(confidence, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(std_probs, bins=20, color="#4c78a8", alpha=0.9)
    axes[0].set_title("Predictive Uncertainty Distribution")
    axes[0].set_xlabel("Std. Dev. across MC samples")
    axes[0].set_ylabel("Count")

    axes[1].scatter(mean_probs, std_probs, s=18, alpha=0.7, color="#f58518")
    axes[1].set_title("Mean Probability vs Uncertainty")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Std. Dev. across MC samples")

    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "uncertainty.png"), dpi=200)
    plt.close(fig)

    uncertainty_payload = {
        "mean_uncertainty": float(np.mean(std_probs)),
        "median_uncertainty": float(np.median(std_probs)),
        "max_uncertainty": float(np.max(std_probs)),
        "mean_confidence": float(np.mean(confidence)),
        "most_uncertain_indices": [int(index) for index in np.argsort(std_probs)[::-1][:5]],
    }
    with open(os.path.join(model_dir, "uncertainty.json"), "w", encoding="utf-8") as handle:
        json.dump(uncertainty_payload, handle, indent=2)


def save_stacking_explainability(model_dir, meta_model):
    meta_importance = np.asarray(meta_model.feature_importances_, dtype=float)
    feature_names = ["xgboost_prob", "fusionnet_prob"]
    payload = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(feature_names, meta_importance)
    ]
    with open(os.path.join(model_dir, "meta_feature_importance.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(feature_names, meta_importance, color=["#4c78a8", "#f58518"])
    axis.set_title("Stacking Meta-Feature Importance")
    axis.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "meta_feature_importance.png"), dpi=200)
    plt.close(fig)


def save_saliency_plots(model_dir, signal_sample_tc, saliency_ct, probability, fs=100):
    signal_sample_tc = np.asarray(signal_sample_tc)
    saliency_ct = np.asarray(saliency_ct)

    if signal_sample_tc.ndim == 1:
        signal_sample_tc = signal_sample_tc[:, np.newaxis]
    if saliency_ct.ndim == 1:
        saliency_ct = saliency_ct[np.newaxis, :]

    channel_names = ["ecg", "edr"]
    channel_count = min(signal_sample_tc.shape[1], saliency_ct.shape[0])

    for channel in range(channel_count):
        name = channel_names[channel] if channel < len(channel_names) else f"channel_{channel}"
        signal_series = signal_sample_tc[:, channel]
        saliency_series = saliency_ct[channel]

        time_axis = np.arange(len(signal_series)) / fs
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        axes[0].plot(time_axis, signal_series, color="#1f77b4", linewidth=1.0)
        axes[0].set_title(f"{name.upper()} Signal (p={probability:.3f})")
        axes[0].set_ylabel("Normalized amplitude")

        axes[1].plot(time_axis, saliency_series, color="#d62728", linewidth=1.0)
        axes[1].fill_between(time_axis, saliency_series, color="#d62728", alpha=0.3)
        axes[1].set_title(f"Temporal Saliency - {name.upper()}")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Normalized saliency")

        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, f"saliency_map_{name}.png"), dpi=200)
        plt.close(fig)


def save_cross_metrics(
    model_label,
    val_metric_name,
    tuned_threshold,
    best_val_score,
    test_metrics,
    base_dir,
    harmonize_level,
    model_dir=None,
):
    if model_dir is None:
        model_dir = ensure_model_dir(model_label, base_dir=base_dir)
    else:
        os.makedirs(model_dir, exist_ok=True)
    payload = {
        "mode": "cross",
        "threshold_metric": val_metric_name,
        "tuned_threshold": float(tuned_threshold),
        "val_score": float(best_val_score),
        "harmonize_preprocessing": harmonize_level != "none",
        "cross_harmonize_level": harmonize_level,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def report_cross_metrics(model_label, threshold_metric, tuned_threshold, best_val_score, test_metrics):
    print(f"\n{model_label} Cross-Dataset Metrics")
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


def evaluate_cross_mode(args, requested):
    if args.no_cross_harmonize:
        harmonize_level = "none"
    elif args.cross_harmonize_level is not None:
        harmonize_level = args.cross_harmonize_level
    else:
        harmonize_level = "light"

    model_dirs = {
        model_key: cross_model_run_dir(model_key, args.cross_signal_mode, harmonize_level)
        for model_key in ["xgboost", "cnn", "chunknet", "fusionnet", "stacking"]
    }

    metadata = None
    for model_key in ["xgboost", "cnn", "chunknet", "fusionnet", "stacking"]:
        metadata_path = artifact_path("metadata.json", base_dir=model_dirs[model_key])
        if os.path.exists(metadata_path):
            metadata = load_json("metadata.json", base_dir=model_dirs[model_key])
            break
    if metadata is None:
        raise FileNotFoundError(
            "No cross-dataset metadata found for the requested configuration. "
            "Run training first for this signal mode and harmonization level."
        )

    cross_fs = int(metadata.get("fs", FS))

    apnea_dir = args.apnea_dir if args.apnea_dir else metadata.get("apnea_dir", DATA_DIR)
    mit_dir = args.mit_dir if args.mit_dir else metadata.get("mit_dir", "mitbih_psg_data")
    mit_val_size = args.mit_val_size if args.mit_val_size is not None else metadata.get("mit_val_size", 0.3)
    random_state = args.random_state if args.random_state is not None else metadata.get("random_state", 42)

    print("=== Loading Cross-Dataset Evaluation Data ===")
    x_train_sig, y_train = load_apnea_ecg_segments_30s(apnea_dir, harmonize_level=harmonize_level)
    x_target_sig, y_target = load_mitbih_psg_segments_30s(mit_dir, harmonize_level=harmonize_level)

    x_val_sig, x_test_sig, y_val, y_test = train_test_split(
        x_target_sig,
        y_target,
        test_size=1.0 - mit_val_size,
        random_state=random_state,
        stratify=y_target,
    )

    summarize_split("Apnea-ECG train", y_train)
    summarize_split("MIT-BIH PSG val", y_val)
    summarize_split("MIT-BIH PSG test", y_test)

    scaler = None
    for model_key in ["xgboost", "cnn", "chunknet", "fusionnet", "stacking"]:
        scaler_path = artifact_path("scaler.joblib", base_dir=model_dirs[model_key])
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            break
    if scaler is None:
        raise FileNotFoundError("No cross-dataset scaler found for the requested configuration.")
    x_val_feat = scaler.transform(build_feature_matrix(x_val_sig))
    x_test_feat = scaler.transform(build_feature_matrix(x_test_sig))

    requested_signal_mode = args.cross_signal_mode
    x_val_sig_n, signal_streams = build_signal_view(x_val_sig, requested_signal_mode, fs=cross_fs)
    x_test_sig_n, _ = build_signal_view(x_test_sig, requested_signal_mode, fs=cross_fs)

    signal_channels = int(x_val_sig_n.shape[2])

    threshold_metric = args.threshold_metric or metadata.get("threshold_metric", "f1")
    summary = {}
    threshold_calibration = {}

    xgb_val_prob = None
    xgb_test_prob = None
    fusion_val_prob = None
    fusion_test_prob = None

    if "xgboost" in requested or "stacking" in requested:
        xgb_model = joblib.load(artifact_path("xgb_model.joblib", base_dir=model_dirs["xgboost"]))
        xgb_val_prob = xgb_model.predict_proba(x_val_feat)[:, 1]
        xgb_test_prob = xgb_model.predict_proba(x_test_feat)[:, 1]

    if "xgboost" in requested:
        threshold_calibration["xgboost"] = compare_threshold_metrics(y_val, xgb_val_prob)
        tuned_threshold, best_val_score = tune_cross_threshold(y_val, xgb_val_prob, threshold_metric)
        test_metrics = compute_cross_metrics(y_test, xgb_test_prob, threshold=tuned_threshold)
        report_cross_metrics("XGBoost", threshold_metric, tuned_threshold, best_val_score, test_metrics)
        save_cross_metrics(
            "XGBoost",
            threshold_metric,
            tuned_threshold,
            best_val_score,
            test_metrics,
            base_dir=model_dirs["xgboost"],
            harmonize_level=harmonize_level,
            model_dir=model_dirs["xgboost"],
        )
        xgb_model_dir = model_dirs["xgboost"]
        save_model_outputs(
            "XGBoost",
            y_test,
            xgb_test_prob,
            (xgb_test_prob >= tuned_threshold).astype(int),
            model_dir=model_dirs["xgboost"],
        )
        feature_names = metadata.get(
            "feature_names",
            [f"feature_{index}" for index in range(len(xgb_model.feature_importances_))],
        )
        save_feature_importance(
            xgb_model_dir,
            feature_names,
            np.asarray(xgb_model.feature_importances_, dtype=float),
        )
        save_shap_outputs(xgb_model_dir, xgb_model, x_test_feat, feature_names, prefix="xgboost")
        summary["xgboost"] = {
            "threshold": float(tuned_threshold),
            "val_score": float(best_val_score),
            **test_metrics,
        }

    if "cnn" in requested:
        cnn_model = CNNBaseline(input_channels=signal_channels).to(DEVICE)
        cnn_model.load_state_dict(torch.load(artifact_path("cnn_model.pt", base_dir=model_dirs["cnn"]), map_location=DEVICE))
        cnn_model.eval()
        cnn_val_prob = predict_probs_signal(cnn_model, x_val_sig_n)
        cnn_test_prob = predict_probs_signal(cnn_model, x_test_sig_n)
        threshold_calibration["cnn"] = compare_threshold_metrics(y_val, cnn_val_prob)
        tuned_threshold, best_val_score = tune_cross_threshold(y_val, cnn_val_prob, threshold_metric)
        test_metrics = compute_cross_metrics(y_test, cnn_test_prob, threshold=tuned_threshold)
        report_cross_metrics("CNN", threshold_metric, tuned_threshold, best_val_score, test_metrics)
        save_cross_metrics(
            "CNN",
            threshold_metric,
            tuned_threshold,
            best_val_score,
            test_metrics,
            base_dir=model_dirs["cnn"],
            harmonize_level=harmonize_level,
            model_dir=model_dirs["cnn"],
        )
        cnn_model_dir = model_dirs["cnn"]
        save_model_outputs(
            "CNN",
            y_test,
            cnn_test_prob,
            (cnn_test_prob >= tuned_threshold).astype(int),
            model_dir=model_dirs["cnn"],
        )
        _, cnn_test_uncertainty, _ = predict_probs_signal_mc_dropout(cnn_model, x_test_sig_n)
        save_uncertainty_outputs(cnn_model_dir, cnn_test_prob, cnn_test_uncertainty)

        cnn_uncertain_index = int(np.argmax(cnn_test_uncertainty))
        cnn_saliency, cnn_saliency_prob = compute_signal_saliency_signal_only(
            cnn_model,
            x_test_sig_n[cnn_uncertain_index].T,
        )
        save_saliency_plots(
            cnn_model_dir,
            x_test_sig_n[cnn_uncertain_index],
            cnn_saliency,
            cnn_saliency_prob,
            fs=cross_fs,
        )
        summary["cnn"] = {
            "threshold": float(tuned_threshold),
            "val_score": float(best_val_score),
            **test_metrics,
        }

    if "chunknet" in requested:
        chunk_model = ChunkCNNLSTM(input_channels=signal_channels).to(DEVICE)
        chunk_model.load_state_dict(torch.load(artifact_path("chunk_model.pt", base_dir=model_dirs["chunknet"]), map_location=DEVICE))
        chunk_model.eval()
        chunk_val_prob = predict_probs_signal(chunk_model, x_val_sig_n)
        chunk_test_prob = predict_probs_signal(chunk_model, x_test_sig_n)
        threshold_calibration["chunknet"] = compare_threshold_metrics(y_val, chunk_val_prob)
        tuned_threshold, best_val_score = tune_cross_threshold(y_val, chunk_val_prob, threshold_metric)
        test_metrics = compute_cross_metrics(y_test, chunk_test_prob, threshold=tuned_threshold)
        report_cross_metrics("ChunkCNNLSTM", threshold_metric, tuned_threshold, best_val_score, test_metrics)
        save_cross_metrics(
            "ChunkCNNLSTM",
            threshold_metric,
            tuned_threshold,
            best_val_score,
            test_metrics,
            base_dir=model_dirs["chunknet"],
            harmonize_level=harmonize_level,
            model_dir=model_dirs["chunknet"],
        )
        chunk_model_dir = model_dirs["chunknet"]
        save_model_outputs(
            "ChunkCNNLSTM",
            y_test,
            chunk_test_prob,
            (chunk_test_prob >= tuned_threshold).astype(int),
            model_dir=model_dirs["chunknet"],
        )
        _, chunk_test_uncertainty, _ = predict_probs_signal_mc_dropout(chunk_model, x_test_sig_n)
        save_uncertainty_outputs(chunk_model_dir, chunk_test_prob, chunk_test_uncertainty)

        chunk_uncertain_index = int(np.argmax(chunk_test_uncertainty))
        chunk_saliency, chunk_saliency_prob = compute_signal_saliency_signal_only(
            chunk_model,
            x_test_sig_n[chunk_uncertain_index].T,
        )
        save_saliency_plots(
            chunk_model_dir,
            x_test_sig_n[chunk_uncertain_index],
            chunk_saliency,
            chunk_saliency_prob,
            fs=cross_fs,
        )
        summary["chunknet"] = {
            "threshold": float(tuned_threshold),
            "val_score": float(best_val_score),
            **test_metrics,
        }

    if "fusionnet" in requested or "stacking" in requested:
        fusion_model = FusionNet(feature_dim=int(metadata["feature_dim"]), input_channels=signal_channels).to(DEVICE)
        fusion_model.load_state_dict(
            torch.load(artifact_path("fusion_model.pt", base_dir=model_dirs["fusionnet"]), map_location=DEVICE)
        )
        fusion_model.eval()
        fusion_val_prob = predict_probs(fusion_model, x_val_sig_n, x_val_feat)
        fusion_test_prob = predict_probs(fusion_model, x_test_sig_n, x_test_feat)
        fusion_test_mean_prob, fusion_test_uncertainty, fusion_mc_samples = predict_probs_mc_dropout(
            fusion_model,
            x_test_sig_n,
            x_test_feat,
        )

    if "fusionnet" in requested:
        threshold_calibration["fusionnet"] = compare_threshold_metrics(y_val, fusion_val_prob)
        tuned_threshold, best_val_score = tune_cross_threshold(y_val, fusion_val_prob, threshold_metric)
        test_metrics = compute_cross_metrics(y_test, fusion_test_prob, threshold=tuned_threshold)
        report_cross_metrics("FusionNet", threshold_metric, tuned_threshold, best_val_score, test_metrics)
        save_cross_metrics(
            "FusionNet",
            threshold_metric,
            tuned_threshold,
            best_val_score,
            test_metrics,
            base_dir=model_dirs["fusionnet"],
            harmonize_level=harmonize_level,
            model_dir=model_dirs["fusionnet"],
        )
        fusion_model_dir = model_dirs["fusionnet"]
        save_model_outputs(
            "FusionNet",
            y_test,
            fusion_test_prob,
            (fusion_test_prob >= tuned_threshold).astype(int),
            model_dir=model_dirs["fusionnet"],
        )
        save_uncertainty_outputs(fusion_model_dir, fusion_test_prob, fusion_test_uncertainty)

        fusion_uncertain_index = int(np.argmax(fusion_test_uncertainty))
        fusion_saliency, fusion_saliency_prob = compute_signal_saliency(
            fusion_model,
            x_test_sig_n[fusion_uncertain_index].T,
            x_test_feat[fusion_uncertain_index],
        )
        save_saliency_plots(
            fusion_model_dir,
            x_test_sig_n[fusion_uncertain_index],
            fusion_saliency,
            fusion_saliency_prob,
            fs=cross_fs,
        )
        summary["fusionnet"] = {
            "threshold": float(tuned_threshold),
            "val_score": float(best_val_score),
            **test_metrics,
        }

    if "stacking" in requested:
        meta_model = joblib.load(artifact_path("stacking_model.joblib", base_dir=model_dirs["stacking"]))
        meta_val = np.column_stack([xgb_val_prob, fusion_val_prob])
        meta_test = np.column_stack([xgb_test_prob, fusion_test_prob])
        meta_val_prob = meta_model.predict_proba(meta_val)[:, 1]
        meta_test_prob = meta_model.predict_proba(meta_test)[:, 1]
        threshold_calibration["stacking"] = compare_threshold_metrics(y_val, meta_val_prob)
        tuned_threshold, best_val_score = tune_cross_threshold(y_val, meta_val_prob, threshold_metric)
        test_metrics = compute_cross_metrics(y_test, meta_test_prob, threshold=tuned_threshold)
        report_cross_metrics("Stacking", threshold_metric, tuned_threshold, best_val_score, test_metrics)
        save_cross_metrics(
            "Stacking",
            threshold_metric,
            tuned_threshold,
            best_val_score,
            test_metrics,
            base_dir=model_dirs["stacking"],
            harmonize_level=harmonize_level,
            model_dir=model_dirs["stacking"],
        )
        stacking_dir = model_dirs["stacking"]
        save_model_outputs(
            "Stacking",
            y_test,
            meta_test_prob,
            (meta_test_prob >= tuned_threshold).astype(int),
            model_dir=model_dirs["stacking"],
        )
        save_stacking_explainability(stacking_dir, meta_model)

        meta_feature_names = ["xgboost_prob", "fusionnet_prob"]
        save_shap_outputs(stacking_dir, meta_model, meta_test, meta_feature_names, prefix="stacking", max_samples=600)

        stacked_mc_inputs = np.stack(
            [xgb_test_prob[np.newaxis, :].repeat(fusion_mc_samples.shape[0], axis=0), fusion_mc_samples],
            axis=2,
        )
        stacked_mc_inputs = stacked_mc_inputs.reshape(-1, 2)
        stacked_mc_probs = meta_model.predict_proba(stacked_mc_inputs)[:, 1].reshape(fusion_mc_samples.shape[0], -1)
        stacking_uncertainty = np.std(stacked_mc_probs, axis=0)
        save_uncertainty_outputs(stacking_dir, meta_test_prob, stacking_uncertainty)
        summary["stacking"] = {
            "threshold": float(tuned_threshold),
            "val_score": float(best_val_score),
            **test_metrics,
        }

    for model_key in requested:
        model_dir = model_dirs[model_key]
        model_summary = summary.get(model_key)
        with open(artifact_path("evaluation_summary.json", base_dir=model_dir), "w", encoding="utf-8") as handle:
            json.dump({model_key: model_summary} if model_summary is not None else {}, handle, indent=2)

        with open(artifact_path("threshold_calibration.json", base_dir=model_dir), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_dataset": "mitbih_psg",
                    "selection_metric": threshold_metric,
                    "signal_mode": requested_signal_mode,
                    "signal_streams": signal_streams,
                    "model": model_key,
                    "thresholds": threshold_calibration.get(model_key),
                },
                handle,
                indent=2,
            )

    print(f"Saved cross-dataset metrics in {CROSS_ARTIFACT_DIR}/")


def main():
    args = parse_args()
    requested = resolve_requested_models(args.models)

    if args.mode == "cross":
        evaluate_cross_mode(args, requested)
        return

    scaler, metadata, test_idx = load_common_artifacts()

    print("=== Loading Data For Evaluation ===")
    x_signal, y = load_segments_and_labels(
        DATA_DIR,
        fs=metadata["fs"],
        window_seconds=metadata["window_seconds"],
        stride_seconds=metadata["stride_seconds"],
    )

    x_features = build_feature_matrix(x_signal)
    x_features = scaler.transform(x_features)
    x_signal = build_ecg_edr_signal(x_signal, fs=metadata["fs"])

    x_test_f = x_features[test_idx]
    y_test = y[test_idx]
    x_signal_test = x_signal[test_idx]
    x_feat_test = x_features[test_idx]

    xgb_model = None
    cnn_model = None
    chunk_model = None
    fusion_model = None
    meta_model = None

    xgb_prob = None
    cnn_prob = None

    if "xgboost" in requested or "stacking" in requested:
        xgb_path = artifact_path("xgb_model.joblib")
        ensure_file_exists(xgb_path)
        xgb_model = joblib.load(xgb_path)

    if "cnn" in requested:
        cnn_path = artifact_path("cnn_model.pt")
        ensure_file_exists(cnn_path)
        signal_channels = int(metadata.get("signal_channels", 1))
        cnn_model = CNNBaseline(input_channels=signal_channels).to(DEVICE)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        cnn_model.eval()

    if "chunknet" in requested:
        chunk_path = artifact_path("chunk_model.pt")
        ensure_file_exists(chunk_path)
        signal_channels = int(metadata.get("signal_channels", 1))
        chunk_model = ChunkCNNLSTM(input_channels=signal_channels).to(DEVICE)
        chunk_model.load_state_dict(torch.load(chunk_path, map_location=DEVICE))
        chunk_model.eval()

    if "fusionnet" in requested or "stacking" in requested:
        fusion_path = artifact_path("fusion_model.pt")
        ensure_file_exists(fusion_path)
        signal_channels = int(metadata.get("signal_channels", 1))
        fusion_model = FusionNet(feature_dim=metadata["feature_dim"], input_channels=signal_channels).to(DEVICE)
        fusion_model.load_state_dict(torch.load(fusion_path, map_location=DEVICE))
        fusion_model.eval()

    if "stacking" in requested:
        stacking_path = artifact_path("stacking_model.joblib")
        ensure_file_exists(stacking_path)
        meta_model = joblib.load(stacking_path)

    if "xgboost" in requested:
        xgb_prob = xgb_model.predict_proba(x_test_f)[:, 1]
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        report_metrics(y_test, xgb_prob, xgb_pred, "XGBoost")
        xgb_model_dir = ensure_model_dir("XGBoost")
        save_model_outputs("XGBoost", y_test, xgb_prob, xgb_pred)
        feature_names = metadata.get(
            "feature_names",
            [f"feature_{index}" for index in range(len(xgb_model.feature_importances_))],
        )
        save_feature_importance(
            xgb_model_dir,
            feature_names,
            np.asarray(xgb_model.feature_importances_, dtype=float),
        )
        save_shap_outputs(xgb_model_dir, xgb_model, x_test_f, feature_names, prefix="xgboost")

    if "cnn" in requested:
        cnn_base_prob, cnn_base_uncertainty, _ = predict_probs_signal_mc_dropout(cnn_model, x_signal_test)
        cnn_base_pred = (cnn_base_prob >= 0.5).astype(int)
        report_metrics(y_test, cnn_base_prob, cnn_base_pred, "CNN")
        cnn_model_dir = ensure_model_dir("CNN")
        save_model_outputs("CNN", y_test, cnn_base_prob, cnn_base_pred)
        save_uncertainty_outputs(cnn_model_dir, cnn_base_prob, cnn_base_uncertainty)

        cnn_uncertain_index = int(np.argmax(cnn_base_uncertainty))
        cnn_saliency, cnn_saliency_prob = compute_signal_saliency_signal_only(
            cnn_model,
            x_signal_test[cnn_uncertain_index].T,
        )
        save_saliency_plots(
            cnn_model_dir,
            x_signal_test[cnn_uncertain_index],
            cnn_saliency,
            cnn_saliency_prob,
            fs=metadata["fs"],
        )

    if "fusionnet" in requested:
        cnn_prob, cnn_uncertainty, _ = predict_probs_mc_dropout(fusion_model, x_signal_test, x_feat_test)
        cnn_pred = (cnn_prob >= 0.5).astype(int)
        report_metrics(y_test, cnn_prob, cnn_pred, "FusionNet")
        cnn_model_dir = ensure_model_dir("FusionNet")
        save_model_outputs("FusionNet", y_test, cnn_prob, cnn_pred)
        save_uncertainty_outputs(cnn_model_dir, cnn_prob, cnn_uncertainty)

        most_uncertain_index = int(np.argmax(cnn_uncertainty))
        saliency, saliency_prob = compute_signal_saliency(
            fusion_model,
            x_signal_test[most_uncertain_index].T,
            x_feat_test[most_uncertain_index],
        )
        save_saliency_plots(
            cnn_model_dir,
            x_signal_test[most_uncertain_index],
            saliency,
            saliency_prob,
            fs=metadata["fs"],
        )

    if "chunknet" in requested:
        chunk_prob, chunk_uncertainty, _ = predict_probs_signal_mc_dropout(chunk_model, x_signal_test)
        chunk_pred = (chunk_prob >= 0.5).astype(int)
        report_metrics(y_test, chunk_prob, chunk_pred, "ChunkCNNLSTM")
        chunk_model_dir = ensure_model_dir("ChunkCNNLSTM")
        save_model_outputs("ChunkCNNLSTM", y_test, chunk_prob, chunk_pred)
        save_uncertainty_outputs(chunk_model_dir, chunk_prob, chunk_uncertainty)

        chunk_uncertain_index = int(np.argmax(chunk_uncertainty))
        chunk_saliency, chunk_saliency_prob = compute_signal_saliency_signal_only(
            chunk_model,
            x_signal_test[chunk_uncertain_index].T,
        )
        save_saliency_plots(
            chunk_model_dir,
            x_signal_test[chunk_uncertain_index],
            chunk_saliency,
            chunk_saliency_prob,
            fs=metadata["fs"],
        )

    if "stacking" in requested:
        if xgb_prob is None:
            xgb_prob = xgb_model.predict_proba(x_test_f)[:, 1]
        if cnn_prob is None:
            cnn_prob, _, fusion_mc_samples = predict_probs_mc_dropout(fusion_model, x_signal_test, x_feat_test)
        else:
            _, _, fusion_mc_samples = predict_probs_mc_dropout(fusion_model, x_signal_test, x_feat_test)

        meta_test = np.column_stack([xgb_prob, cnn_prob])
        meta_prob = meta_model.predict_proba(meta_test)[:, 1]
        meta_pred = (meta_prob >= 0.5).astype(int)
        report_metrics(y_test, meta_prob, meta_pred, "Stacking")
        stacking_dir = ensure_model_dir("Stacking")
        save_model_outputs("Stacking", y_test, meta_prob, meta_pred)
        save_stacking_explainability(stacking_dir, meta_model)

        meta_feature_names = ["xgboost_prob", "fusionnet_prob"]
        save_shap_outputs(stacking_dir, meta_model, meta_test, meta_feature_names, prefix="stacking", max_samples=600)

        stacked_mc_inputs = np.stack(
            [xgb_prob[np.newaxis, :].repeat(fusion_mc_samples.shape[0], axis=0), fusion_mc_samples],
            axis=2,
        )
        stacked_mc_inputs = stacked_mc_inputs.reshape(-1, 2)
        stacked_mc_probs = meta_model.predict_proba(stacked_mc_inputs)[:, 1].reshape(fusion_mc_samples.shape[0], -1)
        stacking_uncertainty = np.std(stacked_mc_probs, axis=0)
        save_uncertainty_outputs(stacking_dir, meta_prob, stacking_uncertainty)

    print("Saved/updated evaluation outputs in artifacts/ for selected models.")


if __name__ == "__main__":
    main()
