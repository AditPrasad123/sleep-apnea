import argparse
import json
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
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
    CNNBaseline,
    DATA_DIR,
    DEVICE,
    FusionNet,
    artifact_path,
    build_ecg_edr_signal,
    build_feature_matrix,
    compute_signal_saliency,
    compute_signal_saliency_signal_only,
    load_array,
    load_json,
    load_segments_and_labels,
    predict_probs_mc_dropout,
    predict_probs_signal,
    predict_probs_signal_mc_dropout,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sleep apnea models selectively.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "xgboost", "cnn", "fusionnet", "stacking"],
        default=["all"],
        help="Models to evaluate. Use one or more: xgboost cnn fusionnet stacking. Default: all",
    )
    return parser.parse_args()


def resolve_requested_models(raw_models):
    if "all" in raw_models:
        return {"xgboost", "cnn", "fusionnet", "stacking"}
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


def ensure_model_dir(model_label):
    model_dir = os.path.join("artifacts", model_label.lower().replace(" ", "_"))
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


def save_model_outputs(model_label, y_true, y_prob, y_pred):
    model_dir = ensure_model_dir(model_label)
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


def main():
    args = parse_args()
    requested = resolve_requested_models(args.models)

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
