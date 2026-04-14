"""Recompute disentangle scores from saved eval JSONs using the full prompt set."""

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

eval_files = {
    'DAS': os.path.join(MODEL_DIR, 'DAS_Continent_evalall.json'),
    'Complement_DAS': os.path.join(MODEL_DIR, 'CompDAS_Continent_evalall.json'),
}

print(f"{'Method':<20} {'Disentangle':>12} {'Isolate':>12} {'Cause':>12}")
print("-" * 60)

for method, path in eval_files.items():
    if not os.path.exists(path):
        print(f"{method:<20} {'MISSING':>12}")
        continue

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

    # Print split counts for debugging.
    n_iso = sum(len(v) for v in attribute_to_iso_tasks.values())
    n_cause = sum(len(v) for v in attribute_to_cause_tasks.values())
    print(f"\n{method}: {n_cause} cause splits, {n_iso} iso splits")

    # Filter out attributes with no matching test splits.
    attribute_to_iso_tasks = {a: ts for a, ts in attribute_to_iso_tasks.items() if ts}
    attribute_to_cause_tasks = {a: ts for a, ts in attribute_to_cause_tasks.items() if ts}

    scores = compute_disentangle_score(data, attribute_to_iso_tasks, attribute_to_cause_tasks)
    print(f"{method:<20} {scores['disentangle']:>12.4f} {scores['isolate']:>12.4f} {scores['cause']:>12.4f}")
