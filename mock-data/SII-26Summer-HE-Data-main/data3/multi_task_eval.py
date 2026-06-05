#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--student-run', default='run.py', help='Path to the original student_package run.py')
parser.add_argument('--runs', default='1')
parser.add_argument('--workers', default='20')
args = parser.parse_args()

root = Path(__file__).resolve().parent
for task in sorted((root / 'tasks').iterdir()):
    train = task / 'train.jsonl'
    test = task / 'test.jsonl'
    print(f'\n=== Running {task.name} ===', flush=True)
    cmd = [sys.executable, args.student_run, '--train', str(train), '--dev', str(test), '--runs', args.runs, '--workers', args.workers]
    subprocess.run(cmd, check=False)
