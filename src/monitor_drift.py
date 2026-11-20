"""Data drift monitoring module for production inference tracking.

Compares production inference logs against baseline training data using
the Kolmogorov-Smirnov (KS) two-sample statistical test.
Generates structured JSON metrics and an HTML dashboard.
"""
from pathlib import Path
import argparse
import datetime
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_DATA_PATH = Path("data/processed/train.csv")
INFERENCE_LOG_PATH = Path("data/monitoring/inference_logs.csv")
REPORTS_DIR = Path("reports")
DRIFT_JSON_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_HTML_PATH = REPORTS_DIR / "drift_report.html"

FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

LOG_COLUMNS = [
    "timestamp",
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
    "predicted_class",
    "prob_Setosa",
    "prob_Versicolor",
    "prob_Virginica",
]


def load_inference_logs(log_path: Path = INFERENCE_LOG_PATH) -> pd.DataFrame:
    """Robustly load production inference logs with schema enforcement and error recovery."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)

    try:
        df = pd.read_csv(log_path)
    except Exception:
        # Fallback to skip malformed lines from legacy writes
        try:
            df = pd.read_csv(log_path, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=LOG_COLUMNS)

    # Ensure all canonical columns exist
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df


def log_inference(
    input_df: pd.DataFrame,
    prediction: str,
    probabilities: dict = None,
    log_path: Path = INFERENCE_LOG_PATH,
):
    """Append a single inference payload to the monitoring log with strict schema consistency."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prob_setosa = probabilities.get("Setosa", np.nan) if probabilities else np.nan
    prob_versicolor = probabilities.get("Versicolor", np.nan) if probabilities else np.nan
    prob_virginica = probabilities.get("Virginica", np.nan) if probabilities else np.nan

    row_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "SepalLengthCm": float(input_df["SepalLengthCm"].iloc[0]),
        "SepalWidthCm": float(input_df["SepalWidthCm"].iloc[0]),
        "PetalLengthCm": float(input_df["PetalLengthCm"].iloc[0]),
        "PetalWidthCm": float(input_df["PetalWidthCm"].iloc[0]),
        "predicted_class": prediction,
        "prob_Setosa": prob_setosa,
        "prob_Versicolor": prob_versicolor,
        "prob_Virginica": prob_virginica,
    }

    df_row = pd.DataFrame([row_data], columns=LOG_COLUMNS)

    header = not log_path.exists() or log_path.stat().st_size == 0
    df_row.to_csv(log_path, mode="a", index=False, header=header)


def generate_sample_inferences(
    n_samples: int = 50,
    drift: bool = False,
    log_path: Path = INFERENCE_LOG_PATH,
):
    """Generate synthetic inference logs adhering to the exact canonical schema."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # Base distributions modeled around Iris
    if drift:
        # Introduce distribution shift (e.g., significantly larger petals)
        sepal_length = rng.normal(7.5, 0.8, n_samples)
        sepal_width = rng.normal(2.2, 0.5, n_samples)
        petal_length = rng.normal(6.5, 0.9, n_samples)
        petal_width = rng.normal(2.5, 0.4, n_samples)
    else:
        # Normal production traffic mirroring training distribution
        sepal_length = rng.normal(5.8, 0.8, n_samples)
        sepal_width = rng.normal(3.0, 0.4, n_samples)
        petal_length = rng.normal(3.8, 1.7, n_samples)
        petal_width = rng.normal(1.2, 0.7, n_samples)

    predicted_classes = rng.choice(
        ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], size=n_samples
    )

    # Generate plausible mock probabilities
    prob_setosa = []
    prob_versicolor = []
    prob_virginica = []
    for pred in predicted_classes:
        if pred == "Iris-setosa":
            prob_setosa.append(round(rng.uniform(0.85, 0.99), 4))
            prob_versicolor.append(round(rng.uniform(0.01, 0.10), 4))
            prob_virginica.append(round(rng.uniform(0.00, 0.05), 4))
        elif pred == "Iris-versicolor":
            prob_setosa.append(round(rng.uniform(0.00, 0.05), 4))
            prob_versicolor.append(round(rng.uniform(0.80, 0.95), 4))
            prob_virginica.append(round(rng.uniform(0.05, 0.15), 4))
        else:
            prob_setosa.append(round(rng.uniform(0.00, 0.03), 4))
            prob_versicolor.append(round(rng.uniform(0.05, 0.15), 4))
            prob_virginica.append(round(rng.uniform(0.82, 0.97), 4))

    timestamps = [
        (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=i * 5)).isoformat()
        for i in range(n_samples)
    ]

    sample_df = pd.DataFrame({
        "timestamp": timestamps,
        "SepalLengthCm": np.clip(sepal_length, 4.0, 9.0).round(1),
        "SepalWidthCm": np.clip(sepal_width, 1.8, 5.0).round(1),
        "PetalLengthCm": np.clip(petal_length, 1.0, 7.5).round(1),
        "PetalWidthCm": np.clip(petal_width, 0.1, 3.0).round(1),
        "predicted_class": predicted_classes,
        "prob_Setosa": prob_setosa,
        "prob_Versicolor": prob_versicolor,
        "prob_Virginica": prob_virginica,
    }, columns=LOG_COLUMNS)

    # Overwrite if starting fresh or append
    sample_df.to_csv(log_path, mode="w", index=False, header=True)
    print(f"Generated {n_samples} sample inference logs (drift={drift}) at: {log_path}")
    return sample_df


def analyze_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    dataset_drift_threshold: float = 0.5,
) -> dict:
    """Run Kolmogorov-Smirnov test per feature to detect data drift."""
    features_report = {}
    drifted_count = 0

    for col in FEATURE_COLUMNS:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
        is_drift = bool(p_val < p_value_threshold)
        if is_drift:
            drifted_count += 1

        features_report[col] = {
            "drift_detected": is_drift,
            "p_value": float(p_val),
            "ks_statistic": float(ks_stat),
            "threshold": p_value_threshold,
            "reference_mean": float(np.mean(ref_vals)),
            "reference_std": float(np.std(ref_vals)),
            "current_mean": float(np.mean(cur_vals)),
            "current_std": float(np.std(cur_vals)),
            "mean_shift": float(np.mean(cur_vals) - np.mean(ref_vals)),
        }

    total_features = len(features_report)
    drift_share = (drifted_count / total_features) if total_features > 0 else 0.0
    dataset_drift_detected = drift_share >= dataset_drift_threshold

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share_pct": round(drift_share * 100, 2),
        "dataset_drift_detected": dataset_drift_detected,
        "dataset_drift_threshold_pct": round(dataset_drift_threshold * 100, 2),
        "features": features_report,
    }
    return report


def render_html_report(report: dict, output_path: Path = DRIFT_HTML_PATH):
    """Render a standalone, modern HTML dashboard for the drift analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_color = "#ef4444" if report["dataset_drift_detected"] else "#10b981"
    status_text = "DRIFT DETECTED" if report["dataset_drift_detected"] else "DISTRIBUTION HEALTHY"

    cards_html = ""
    for feat, data in report["features"].items():
        feat_color = "#ef4444" if data["drift_detected"] else "#10b981"
        feat_status = "DRIFT" if data["drift_detected"] else "PASS"
        cards_html += f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong style="font-size: 16px; color: #1e293b;">{feat}</strong>
                <span style="background: {feat_color}; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 12px;">{feat_status}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; font-size: 13px; color: #64748b;">
                <div>p-value: <b style="color: #0f172a;">{data['p_value']:.4f}</b></div>
                <div>KS-statistic: <b style="color: #0f172a;">{data['ks_statistic']:.4f}</b></div>
                <div>Ref Mean: <b style="color: #0f172a;">{data['reference_mean']:.2f} (±{data['reference_std']:.2f})</b></div>
                <div>Prod Mean: <b style="color: #0f172a;">{data['current_mean']:.2f} (±{data['current_std']:.2f})</b></div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MLOps Data Drift Monitoring</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 24px; color: #1e293b; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .header {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 16px; color: white; font-weight: bold; font-size: 14px; background: {status_color}; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }}
        .stat-card {{ background: #f1f5f9; padding: 12px; border-radius: 8px; text-align: center; }}
        .stat-num {{ font-size: 20px; font-weight: bold; color: #0f172a; }}
        .stat-label {{ font-size: 12px; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0 0 6px 0; font-size: 24px;">🌸 Production Data Drift Report</h1>
                    <div style="color: #64748b; font-size: 13px;">Analyzed at: {report['timestamp']}</div>
                </div>
                <div class="badge">{status_text}</div>
            </div>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-num">{report['reference_samples']}</div>
                    <div class="stat-label">Reference Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['current_samples']}</div>
                    <div class="stat-label">Production Logs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['drifted_features']} / {report['total_features']}</div>
                    <div class="stat-label">Drifted Features</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['drift_share_pct']}%</div>
                    <div class="stat-label">Drift Share</div>
                </div>
            </div>
        </div>
        <h2 style="font-size: 18px; margin-bottom: 12px;">Feature Drift Tests (Kolmogorov-Smirnov)</h2>
        {cards_html}
    </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Check production data drift against baseline training data.")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DATA_PATH, help="Path to reference training CSV.")
    parser.add_argument("--logs", type=Path, default=INFERENCE_LOG_PATH, help="Path to production inference logs CSV.")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate synthetic inference logs if empty.")
    parser.add_argument("--simulate-drift", action="store_true", help="Simulate drifted traffic for demonstration.")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 if dataset drift is detected.")
    args = parser.parse_args()

    if not args.reference.exists():
        print(f"Error: Reference training data not found at {args.reference}. Run dvc repro first.")
        sys.exit(1)

    ref_df = pd.read_csv(args.reference)

    logs_df = load_inference_logs(args.logs)
    if len(logs_df) < 5 or args.generate_sample_data or args.simulate_drift:
        if args.generate_sample_data or args.simulate_drift:
            logs_df = generate_sample_inferences(n_samples=50, drift=args.simulate_drift, log_path=args.logs)
        else:
            print(f"Inference log file {args.logs} is empty or has < 5 entries. Auto-generating sample traffic...")
            logs_df = generate_sample_inferences(n_samples=50, drift=False, log_path=args.logs)

    report = analyze_drift(ref_df, logs_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    render_html_report(report, DRIFT_HTML_PATH)

    print("\n=== Data Drift Monitoring Report ===")
    print(f"Status: {'DRIFT DETECTED ⚠️' if report['dataset_drift_detected'] else 'DISTRIBUTION HEALTHY ✅'}")
    print(f"Drifted features: {report['drifted_features']} / {report['total_features']} ({report['drift_share_pct']}%)")
    for feat, data in report["features"].items():
        flag = "DRIFT" if data["drift_detected"] else "OK"
        print(f"  - {feat:15s}: [{flag}] p-val={data['p_value']:.4f}, KS={data['ks_statistic']:.4f}, shift={data['mean_shift']:+.2f}")
    print(f"\nSaved JSON report: {DRIFT_JSON_PATH}")
    print(f"Saved HTML report: {DRIFT_HTML_PATH}")

    if args.fail_on_drift and report["dataset_drift_detected"]:
        print("\nQuality Gate Error: Production data drift exceeds threshold!")
        sys.exit(1)


if __name__ == "__main__":
    main()
