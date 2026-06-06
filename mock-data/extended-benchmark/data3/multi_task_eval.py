#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--runner', default='run.py', help='Path to the evaluation runner script')
parser.add_argument('--runs', default='1')
parser.add_argument('--workers', default='20')
args = parser.parse_args()

root = Path(__file__).resolve().parent
for task in sorted((root / 'tasks').iterdir()):
    train = task / 'train.jsonl'
    test = task / 'test.jsonl'
    print(f'\n=== Running {task.name} ===', flush=True)
    cmd = [sys.executable, args.runner, '--train', str(train), '--dev', str(test), '--runs', args.runs, '--workers', args.workers]
    subprocess.run(cmd, check=False)
