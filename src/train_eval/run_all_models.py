import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.train_eval.common import EXPERIMENT_DIR_ENV, create_experiment_root
from src.utils.helpers import load_config


def run_command(command, stage_name, environment):
    """Run one stage and measure its execution time."""
    print("\n" + "=" * 100)
    print(f"Running: {stage_name}")
    print(f"Command: {' '.join(command)}")
    print("=" * 100)

    start_time = time.perf_counter()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, text=True, env=environment)
    end_time = time.perf_counter()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    duration_seconds = end_time - start_time
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"Finished: {stage_name} | {status} | {duration_seconds / 60:.2f} minutes")

    return {
        "stage": stage_name,
        "command": " ".join(command),
        "status": status,
        "return_code": result.returncode,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60,
    }


def main():
    """Run all models and their shared post-processing in one experiment folder."""
    config = load_config("config.yaml")
    experiment_dir = create_experiment_root(config).resolve()
    environment = os.environ.copy()
    environment[EXPERIMENT_DIR_ENV] = str(experiment_dir)

    print(f"Experiment directory: {experiment_dir}")
    python = sys.executable
    stages = [
        ("Raw EEGNet-like CNN", "src/train_eval/run_cnn_eegnetlike.py"),
        ("Features1D CNN", "src/train_eval/run_cnn_features1d.py"),
        ("Features2D CNN", "src/train_eval/run_cnn_features2d.py"),
        ("Hybrid Raw + Features CNN", "src/train_eval/run_cnn_hybrid.py"),
    ]
    rows = []
    started = time.perf_counter()

    for stage_name, script in stages:
        if not Path(script).is_file():
            rows.append({"stage": stage_name, "command": f"{python} {script}", "status": "SKIPPED_SCRIPT_NOT_FOUND",
                         "return_code": None, "start_datetime": None, "end_datetime": None,
                         "duration_seconds": None, "duration_minutes": None})
            break
        row = run_command([python, script], stage_name, environment)
        rows.append(row)
        if row["status"] != "SUCCESS":
            print("Stopping because a model stage failed.")
            break
    else:
        for stage_name, script in [
            ("Model comparison analysis", "src/train_eval/analyze_model_results.py"),
            ("Statistical tests", "src/train_eval/run_statistical_tests.py"),
        ]:
            row = run_command([python, script], stage_name, environment)
            rows.append(row)
            if row["status"] != "SUCCESS":
                break

    timing_csv_path = experiment_dir / "experiment_timings" / "model_runtime_summary.csv"
    with open(timing_csv_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["stage", "command", "status", "return_code", "start_datetime",
                                                     "end_datetime", "duration_seconds", "duration_minutes"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print("EXPERIMENT FINISHED")
    print("=" * 100)
    print(f"Total duration: {(time.perf_counter() - started) / 60:.2f} minutes")
    print(f"Timing summary: {timing_csv_path}")
    print(f"All outputs: {experiment_dir}")


if __name__ == "__main__":
    main()
