import argparse
import subprocess
import sys
from dataclasses import dataclass


SIGNAL_MODES = ("ecg", "edr", "ecg_edr")
HARMONIZE_LEVELS = ("none", "light", "full")


@dataclass
class ComboResult:
    signal_mode: str
    harmonize_level: str
    train_ok: bool
    eval_ok: bool
    error: str = ""


def build_common_args(args):
    common = [
        "--mode",
        "cross",
        "--models",
        "all",
        "--threshold-metric",
        args.threshold_metric,
        "--mit-val-size",
        str(args.mit_val_size),
        "--random-state",
        str(args.random_state),
        "--apnea-dir",
        args.apnea_dir,
        "--mit-dir",
        args.mit_dir,
        "--few-shot-mit-frac",
        str(args.few_shot_mit_frac),
        "--few-shot-epochs",
        str(args.few_shot_epochs),
        "--few-shot-lr",
        str(args.few_shot_lr),
    ]
    
    return common


def run_command(command, dry_run=False):
    print(" ".join(command))
    if dry_run:
        return True, ""

    try:
        subprocess.run(command, check=True)
        return True, ""
    except subprocess.CalledProcessError as error:
        return False, str(error)


def run_suite(args):
    common = build_common_args(args)
    results = []

    for signal_mode in args.signal_modes:
        for harmonize_level in args.harmonize_levels:
            print("=" * 80)
            print(f"Running combo: signal_mode={signal_mode}, harmonize_level={harmonize_level}")

            combo_flags = [
                "--cross-signal-mode",
                signal_mode,
                "--cross-harmonize-level",
                harmonize_level,
            ]

            train_ok = True
            eval_ok = True
            error_msg = ""

            if not args.evaluate_only:
                train_cmd = [sys.executable, "train.py", *common, *combo_flags]
                train_ok, error_msg = run_command(train_cmd, dry_run=args.dry_run)

            if train_ok and not args.train_only:
                eval_cmd = [sys.executable, "evaluate.py", *common, *combo_flags]
                eval_ok, error_msg = run_command(eval_cmd, dry_run=args.dry_run)
            elif not train_ok:
                eval_ok = False if not args.train_only else True

            results.append(
                ComboResult(
                    signal_mode=signal_mode,
                    harmonize_level=harmonize_level,
                    train_ok=train_ok,
                    eval_ok=eval_ok,
                    error=error_msg,
                )
            )

            if not train_ok or not eval_ok:
                print(f"FAILED: signal_mode={signal_mode}, harmonize_level={harmonize_level}")
                print(error_msg)
                if args.fail_fast:
                    return results

    return results


def print_summary(results, train_only=False, evaluate_only=False):
    print("\n" + "#" * 80)
    print("Cross-Ablation Suite Summary")
    print("#" * 80)

    passed = 0
    total = len(results)

    for item in results:
        if train_only:
            ok = item.train_ok
            status = "PASS" if ok else "FAIL"
        elif evaluate_only:
            ok = item.eval_ok
            status = "PASS" if ok else "FAIL"
        else:
            ok = item.train_ok and item.eval_ok
            status = "PASS" if ok else "FAIL"

        if ok:
            passed += 1

        print(
            f"[{status}] signal_mode={item.signal_mode}, "
            f"harmonize_level={item.harmonize_level}"
        )

    print("-" * 80)
    print(f"Completed: {passed}/{total}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full cross-dataset ablation suite for all models over all signal/harmonization modes."
    )
    parser.add_argument("--apnea-dir", default="apnea_data", help="Path to Apnea-ECG directory.")
    parser.add_argument("--mit-dir", default="mitbih_psg_data", help="Path to MIT-BIH PSG directory.")
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "balanced_accuracy", "mcc"],
        default="balanced_accuracy",
        help="Metric used for threshold tuning in cross evaluation.",
    )
    parser.add_argument("--mit-val-size", type=float, default=0.3, help="MIT val split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-only", action="store_true", help="Run only train.py for each combo.")
    parser.add_argument("--evaluate-only", action="store_true", help="Run only evaluate.py for each combo.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failed combo.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--harmonize-levels",
        nargs="+",
        choices=HARMONIZE_LEVELS,
        default=list(HARMONIZE_LEVELS),
        help="One or more harmonization levels to run. Default: none light full.",
    )
    parser.add_argument(
        "--signal-modes",
        nargs="+",
        choices=SIGNAL_MODES,
        default=list(SIGNAL_MODES),
        help="One or more signal modes to run. Default: ecg edr ecg_edr.",
    )
    parser.add_argument(
        "--few-shot-mit-frac",
        type=float,
        default=0.0,
        help="Fraction of MIT validation split to use for few-shot adaptation. Default: 0.0 (disabled)",
    )
    parser.add_argument(
        "--few-shot-epochs",
        type=int,
        default=5,
        help="Number of epochs for few-shot fine-tuning. Default: 5",
    )
    parser.add_argument(
        "--few-shot-lr",
        type=float,
        default=1e-4,
        help="Learning rate for few-shot fine-tuning. Default: 1e-4",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train_only and args.evaluate_only:
        raise ValueError("Choose at most one of --train-only or --evaluate-only.")

    results = run_suite(args)
    print_summary(results, train_only=args.train_only, evaluate_only=args.evaluate_only)


if __name__ == "__main__":
    main()
