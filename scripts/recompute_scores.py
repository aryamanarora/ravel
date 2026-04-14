"""Recompute disentangle scores from saved eval JSONs using the full prompt set."""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from utils.metric_utils import compute_disentangle_score

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BASE_DATA_DIR = os.path.join(REPO_DIR, 'run_data', 'base')
MODEL_DIR = os.path.join(REPO_DIR, 'run_models')
TARGET_ATTR = 'Continent'

attr_to_prompts = json.load(
    open(os.path.join(BASE_DATA_DIR, f'ravel_city_attribute_to_prompts.json')))

# Auto-discover all eval JSONs.
eval_files = sorted(glob.glob(os.path.join(MODEL_DIR, '*_evalall.json')))

print(f"{'Method':<25} {'Disentangle':>12} {'Isolate':>12} {'Cause':>12}")
print("-" * 65)

for path in eval_files:
    method = os.path.basename(path).replace('_evalall.json', '')
    data = json.load(open(path))
    test_keys = set(data.keys())

    attribute_to_iso_tasks = {
        a: [p + '-test' for p in ps if p + '-test' in test_keys]
        for a, ps in attr_to_prompts.items() if a != TARGET_ATTR
    }
    attribute_to_cause_tasks = {
        a: [p + '-test' for p in ps if p + '-test' in test_keys]
        for a, ps in attr_to_prompts.items() if a == TARGET_ATTR
    }

    # Filter out attributes with no matching test splits.
    attribute_to_iso_tasks = {a: ts for a, ts in attribute_to_iso_tasks.items() if ts}
    attribute_to_cause_tasks = {a: ts for a, ts in attribute_to_cause_tasks.items() if ts}

    scores = compute_disentangle_score(data, attribute_to_iso_tasks, attribute_to_cause_tasks)
    print(f"{method:<25} {scores['disentangle']:>12.4f} {scores['isolate']:>12.4f} {scores['cause']:>12.4f}")
