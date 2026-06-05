"""Run the extended robustness benchmark suite.

This script evaluates the harness on community/mock datasets stored under
mock-data/. It keeps the official run.py entrypoint unchanged and only
orchestrates repeated local evaluations.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "mock-data" / "SII-26Summer-HE-Data-main"
RUN_PY = PROJECT_ROOT / "run.py"

TASKS = [
    ("Data1 - official dev distribution", "data1/train_dev.jsonl", "data1/test_dev.jsonl"),
    ("Data1 - multiple-choice QA", "data1/train_mcq.jsonl", "data1/test_mcq.jsonl"),
    ("Data1 - out-of-domain labels", "data1/train_ood.jsonl", "data1/test_ood.jsonl"),
    ("Data1 - large label space OOD", "data1/train_ood_more_lables.jsonl", "data1/test_ood_more_lables.jsonl"),
    ("Data1 - Chinese classification", "data1/train_dev_chinese.jsonl", "data1/test_dev_chinese.jsonl"),
    ("Data1 - Chinese MCQ", "data1/train_mcq_chinese.jsonl", "data1/test_mcq_chinese.jsonl"),
    ("Data2 - base task", "data2/task1/train.jsonl", "data2/task1/test.jsonl"),
    ("Data2 - education", "data2/task2_education/train.jsonl", "data2/task2_education/test.jsonl"),
    ("Data2 - restaurant", "data2/task2_restaurant/train.jsonl", "data2/task2_restaurant/test.jsonl"),
    ("Data2 - tech support", "data2/task2_techsupport/train.jsonl", "data2/task2_techsupport/test.jsonl"),
    ("Data2 - complex multi-choice", "data2/task3/train.jsonl", "data2/task3/test.jsonl"),
    ("Data3 - finance domain", "data3/domain_splits/fin/train.jsonl", "data3/domain_splits/fin/test.jsonl"),
    ("Data3 - HR domain", "data3/domain_splits/hr/train.jsonl", "data3/domain_splits/hr/test.jsonl"),
]


def run_task(train_path: Path, test_path: Path, workers: int = 15) -> tuple[str, str]:
    cmd = [
        sys.executable,
        str(RUN_PY),
        "--train",
        str(train_path),
        "--dev",
        str(test_path),
        "--runs",
        "1",
        "--workers",
        str(workers),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    if result.returncode != 0:
        return "Error", f"exit={result.returncode}"

    acc_match = re.search(r"平均准确率:\s*([\d.]+)%", result.stdout)
    time_match = re.search(r"总耗时:\s*([\d.]+)s", result.stdout)
    acc = acc_match.group(1) + "%" if acc_match else "N/A"
    elapsed = time_match.group(1) + "s" if time_match else "N/A"
    return acc, elapsed


def main() -> None:
    print("=" * 78)
    print("Harness Engineering Extended Benchmark")
    print("=" * 78)
    print(f"| {'Task':<40} | {'Accuracy':<10} | {'Time':<10} |")
    print("|" + "-" * 42 + "|" + "-" * 12 + "|" + "-" * 12 + "|")

    start_time = time.time()
    for name, train_rel, test_rel in TASKS:
        train_path = BASE_DIR / train_rel
        test_path = BASE_DIR / test_rel
        if not train_path.exists() or not test_path.exists():
            print(f"| {name:<40} | {'missing':<10} | {'-':<10} |")
            continue

        acc, elapsed = run_task(train_path, test_path)
        print(f"| {name:<40} | {acc:<10} | {elapsed:<10} |")

    total_time = time.time() - start_time
    print("=" * 78)
    print(f"Finished in {total_time / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
