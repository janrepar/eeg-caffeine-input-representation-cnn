import subprocess
import sys
import time
import csv
from pathlib import Path
from datetime import datetime


def run_command(command, model_name):
    """
    Runs one model script and measures execution time.
    """

    print("\n" + "=" * 100)
    print(f"Running model: {model_name}")
    print(f"Command: {' '.join(command)}")
    print("=" * 100)

    start_time = time.perf_counter()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = subprocess.run(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    end_time = time.perf_counter()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60

    status = "SUCCESS" if result.returncode == 0 else "FAILED"

    print("\n" + "-" * 100)
    print(f"Finished model: {model_name}")
    print(f"Status: {status}")
    print(f"Duration: {duration_minutes:.2f} minutes")
    print("-" * 100)

    return {
        "model_name": model_name,
        "command": " ".join(command),
        "status": status,
        "return_code": result.returncode,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_minutes,
    }


def main():
    """
    Runs all experiment scripts one after another.

    Before running this, set in config.yaml:

    experiment: analysis_type: caffeine_before_vs_after
    """

    output_dir = Path("outputs/experiment_timings")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    timing_csv_path = output_dir / f"model_runtime_summary_{timestamp}.csv"

    python_executable = sys.executable

    experiments = [
        {
            "model_name": "Raw EEGNet-like CNN",
            "script": "src/train_eval/run_cnn_eegnetlike.py",
        },
        {
            "model_name": "Features1D CNN",
            "script": "src/train_eval/run_cnn_features1d.py",
        },
        {
            "model_name": "Features2D CNN",
            "script": "src/train_eval/run_cnn_features2d.py",
        },
        {
            "model_name": "Hybrid Raw + Features CNN",
            "script": "src/train_eval/run_cnn_hybrid.py",
        },
    ]

    results = []

    total_start_time = time.perf_counter()

    for experiment in experiments:
        model_name = experiment["model_name"]
        script = experiment["script"]

        script_path = Path(script)

        if not script_path.exists():
            print(f"\nSkipping {model_name}. Script not found: {script}")
            results.append(
                {
                    "model_name": model_name,
                    "command": f"{python_executable} {script}",
                    "status": "SKIPPED_SCRIPT_NOT_FOUND",
                    "return_code": None,
                    "start_datetime": None,
                    "end_datetime": None,
                    "duration_seconds": None,
                    "duration_minutes": None,
                }
            )
            continue

        command = [python_executable, script]

        result_row = run_command(command, model_name)
        results.append(result_row)

        if result_row["status"] == "FAILED":
            print(f"\nModel failed: {model_name}")
            print("Stopping execution to avoid wasting time.")
            break

    total_end_time = time.perf_counter()
    total_duration_seconds = total_end_time - total_start_time
    total_duration_minutes = total_duration_seconds / 60

    with open(timing_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_name",
            "command",
            "status",
            "return_code",
            "start_datetime",
            "end_datetime",
            "duration_seconds",
            "duration_minutes",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print("\n" + "=" * 100)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 100)
    print(f"Total duration: {total_duration_minutes:.2f} minutes")
    print(f"Timing CSV saved to: {timing_csv_path}")


if __name__ == "__main__":
    main()