"""End-to-end: generate RAVEL instance for Llama-2-7B, train DAS + complement DAS on Continent, evaluate.

Follows the pipeline from the TinyLlama create-instance Colab, adapted for Llama-2-7B.
"""

import collections
import json
import os
import random
import re
import sys
import tarfile

import datasets
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src and scripts to path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, SCRIPT_DIR)

from datasets import Dataset
from methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
from utils.generation_utils import generate_batched
from utils.generate_ravel_instance import RAVELMetadata, gen_context_test_split, gen_entity_test_split
from utils.intervention_utils import eval_with_interventions, remove_all_forward_hooks
from utils.metric_utils import compute_metrics, compute_disentangle_score


def _get_inv(v):
    """Unwrap intervention value (handles both old tuple and new direct formats)."""
    return v[0] if isinstance(v, (list, tuple)) else v

# ─── Config ───
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
INSTANCE = "llama2-7b"
ENTITY_TYPE = "city"
TARGET_ATTR = "Continent"
INV_LAYER = 15
INV_DIM = 128
INPUT_MAX_LEN = 48
MAX_OUTPUT_TOKENS = 3
TRAINING_EPOCH = 3
TRAINING_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
CAUSE_TASK_SAMPLE_SIZE = 20000
LR = 1e-4
COMPLEMENT_LOSS_COEFF = 10.0

REPO_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(REPO_DIR, 'run_data')
BASE_DATA_DIR = os.path.join(DATA_DIR, 'base')
INSTANCE_DIR = os.path.join(DATA_DIR, INSTANCE)
MODEL_DIR = os.path.join(REPO_DIR, 'run_models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_TYPES = datasets.Features({
    "input": datasets.Value("string"),
    "label": datasets.Value("string"),
    "source_input": datasets.Value("string"),
    "source_label": datasets.Value("string"),
    "inv_label": datasets.Value("string"),
    'split': datasets.Value("string"),
    'source_split': datasets.Value("string"),
    'entity': datasets.Value("string"),
    'source_entity': datasets.Value("string"),
})


# ─── Label extraction (from Colab) ───

def extract_label(text):
    """Extract first word/phrase from model output. Rules from the Colab."""
    tokens = re.split(r'(["]|[.,;]\s|\n| \(|\sand)', text + ' ')
    x = tokens[0]
    digit_match = re.search(r'\.\d\d', x)
    if digit_match:
        x = x[:digit_match.span(0)[1]]
    gender_match = re.match(r'\s?(his|her|himself|herself|she|he)[^\w]', x)
    if gender_match:
        x = x[:gender_match.span(1)[1]]
    if not x.strip():
        x = ' '.join(text.split(' ')[:2]).rstrip('.,"\n')
    if not x.strip():
        return text
    return x


def get_first_token(x):
    return re.split(r'[^\w\+\-]', x.strip(), re.UNICODE)[0]


def filter_inv_example(base_output, inv_output):
    different_outputs = get_first_token(base_output) != get_first_token(inv_output)
    valid_outputs = (
        re.fullmatch(r'\s?[a-z0-9.:\-+]+', extract_label(base_output), re.IGNORECASE) and
        re.fullmatch(r'\s?[a-z0-9.:\-+]+', extract_label(inv_output), re.IGNORECASE))
    return len(inv_output) > 0 and valid_outputs and different_outputs


# ─── Step 1: Extract base data from data.tgz ───

def extract_base_data():
    if os.path.exists(BASE_DATA_DIR) and os.listdir(BASE_DATA_DIR):
        print("Base data already extracted.")
        return
    tgz_path = os.path.join(REPO_DIR, 'data.tgz')
    print(f"Extracting base data from {tgz_path}...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(DATA_DIR)
    # data.tgz extracts to data/ subdirectory, move contents to base/
    extracted_dir = os.path.join(DATA_DIR, 'data')
    if os.path.exists(extracted_dir):
        os.rename(extracted_dir, BASE_DATA_DIR)
    print(f"Base data extracted to {BASE_DATA_DIR}")


# ─── Step 2: Load base data ───

def load_base_data():
    extract_base_data()

    entity_attributes = json.load(
        open(os.path.join(BASE_DATA_DIR, f'ravel_{ENTITY_TYPE}_entity_attributes.json')))
    attribute_prompts = json.load(
        open(os.path.join(BASE_DATA_DIR, f'ravel_{ENTITY_TYPE}_attribute_to_prompts.json')))
    prompt_splits = json.load(
        open(os.path.join(BASE_DATA_DIR, f'ravel_{ENTITY_TYPE}_prompt_to_split.json')))
    entity_splits = json.load(
        open(os.path.join(BASE_DATA_DIR, f'ravel_{ENTITY_TYPE}_entity_to_split.json')))
    wiki_prompt_splits = json.load(
        open(os.path.join(BASE_DATA_DIR, f'wikipedia_{ENTITY_TYPE}_entity_prompts.json')))

    print(f'#entities={len(entity_attributes)}, #prompt_templates={sum(map(len, attribute_prompts.values()))}')

    # Build prompt -> metadata mapping.
    prompts_to_metadata = {
        t % x: {'entity': x, 'attr': a, 'template': t}
        for x in entity_attributes
        for a, ts in attribute_prompts.items()
        for t in ts
    }
    print(f'Total prompt x entity: {len(prompts_to_metadata)}')

    return (entity_attributes, attribute_prompts, prompt_splits,
            entity_splits, wiki_prompt_splits, prompts_to_metadata)


# ─── Step 3: Generate model outputs + behavioral test ───

def generate_outputs_and_filter(model, tokenizer, entity_attributes, attribute_prompts,
                                 prompt_splits, prompts_to_metadata):
    """Run model on all prompts, then do behavioral test to filter entities/templates."""
    cache_path = os.path.join(INSTANCE_DIR, f'{INSTANCE}_city_prompt_to_output.json')
    if os.path.exists(cache_path):
        print(f"Loading cached model outputs from {cache_path}")
        prompt_to_output = json.load(open(cache_path))
    else:
        print("Generating model outputs...")
        outputs = generate_batched(
            model, tokenizer,
            list(prompts_to_metadata.keys()),
            prompt_max_length=INPUT_MAX_LEN,
            max_new_tokens=8,
            batch_size=64)
        prompt_to_output = {k: v[len(k):] for k, v in outputs}
        json.dump(prompt_to_output, open(cache_path, 'w'), ensure_ascii=False)
        print(f"Saved {len(prompt_to_output)} outputs to {cache_path}")

    # ─── Behavioral test: filter entities and templates by accuracy ───
    print("\nRunning behavioral test...")
    sorted_entity = sorted(set(v['entity'] for v in prompts_to_metadata.values()))
    sorted_template = sorted(set(v['template'] for v in prompts_to_metadata.values()))
    stats = np.zeros([len(sorted_entity), len(sorted_template)])

    for p, out in prompt_to_output.items():
        if p not in prompts_to_metadata:
            continue
        meta = prompts_to_metadata[p]
        entity, attr = meta['entity'], meta['attr']
        label = entity_attributes[entity].get(attr, '')
        if not label:
            continue
        norm_label = label.lower()
        norm_out = out.split('"')[0].strip(' "').replace('\\/', '/').lower()
        if len(norm_label) < len(norm_out):
            correct = int(norm_out.startswith(norm_label))
        else:
            correct = int(norm_label.startswith(norm_out))
        # Latitude/Longitude exceptions.
        if re.search('coord|"lat"|"long"|latitude|coordinates|longitude', p):
            try:
                correct = int(abs(float(norm_label.strip('-\u2212')) - float(re.findall(r'\d+', norm_out)[0])) <= 2)
            except:
                correct = 0
        # Country exceptions.
        if re.search('United States|United Kingdom', label):
            norm_label2 = label.strip().replace('the ', '')
            norm_out2 = out.strip().replace('the ', '')
            correct = int(norm_out2.startswith(norm_label2) or norm_out2.startswith('England'))
        if re.search('South Korea', label):
            correct = int(norm_out.startswith('korea') or norm_out.startswith('south korea'))
        if re.search('North America', label):
            correct = norm_label in norm_out or norm_out == 'na' or norm_out.startswith('america')
        if re.search('Mandarin', label):
            correct = norm_out in norm_label or norm_out == 'chinese'
        if re.search('language', p) and ',' in norm_label:
            correct = any(lang in norm_out for lang in norm_label.split(','))
        stats[sorted_entity.index(entity), sorted_template.index(meta['template'])] += int(correct)

    # Keep top 400 entities and top templates.
    kept_entity_index = np.argsort(stats.sum(axis=1))[-400:]
    KEPT_ENTITY = [sorted_entity[i] for i in kept_entity_index]
    topk_template_index = set(np.argsort(stats.sum(axis=0))[-200:])
    kept_template_index = []
    KEPT_ATTR_TO_PROMPT_AND_SPLIT = {}
    for attr in attribute_prompts:
        attr_indices = [sorted_template.index(t) for t in attribute_prompts[attr]]
        per_attr_kept = sorted(attr_indices, key=lambda i: stats[:, i].sum())[-12:][::-1]
        per_attr_kept = [x for i, x in enumerate(per_attr_kept)
                         if x in topk_template_index or i < 4]
        kept_template_index.extend(per_attr_kept)
        KEPT_ATTR_TO_PROMPT_AND_SPLIT[attr] = {
            sorted_template[i]: prompt_splits[sorted_template[i]]
            for i in per_attr_kept
        }

    avg_acc = 100 * stats[:, kept_template_index][kept_entity_index, :].sum() / (
        len(kept_entity_index) * len(kept_template_index))
    print(f'Kept {len(KEPT_ENTITY)} entities, {len(kept_template_index)} templates, avg accuracy: {avg_acc:.2f}%')

    return prompt_to_output, KEPT_ENTITY, KEPT_ATTR_TO_PROMPT_AND_SPLIT


# ─── Step 4: Create RAVEL instance ───

def create_ravel_instance(model, tokenizer, prompt_to_output,
                          kept_entity, kept_attr_to_prompt_and_split,
                          entity_splits, wiki_prompt_splits, attribute_prompts):
    train_path = os.path.join(INSTANCE_DIR, f'{INSTANCE}_{ENTITY_TYPE}_train.json')
    context_test_path = os.path.join(INSTANCE_DIR, f'{INSTANCE}_{ENTITY_TYPE}_context_test.json')
    entity_test_path = os.path.join(INSTANCE_DIR, f'{INSTANCE}_{ENTITY_TYPE}_entity_test.json')

    if all(os.path.exists(p) for p in [train_path, context_test_path, entity_test_path]):
        print("Loading cached RAVEL instance data...")
        split_to_raw = json.load(open(train_path))
        split_to_raw.update(json.load(open(context_test_path)))
        split_to_raw.update(json.load(open(entity_test_path)))
        return split_to_raw, kept_attr_to_prompt_and_split

    # Filtered entity splits.
    kept_entity_splits = {e: entity_splits[e] for e in kept_entity}
    # Filtered prompt splits.
    kept_prompt_splits = {
        k: (a, v) for a, d in kept_attr_to_prompt_and_split.items()
        for k, v in d.items() if k.count('%') == 1
    }
    for prompt in wiki_prompt_splits:
        kept_prompt_splits[prompt] = ('Other', wiki_prompt_splits[prompt]['split'])

    # Filter attr_to_prompt_and_split to single-%s templates.
    kept_attr_to_prompt_and_split = {
        k: {p: v for p, v in d.items() if p.count('%') == 1}
        for k, d in kept_attr_to_prompt_and_split.items()
    }

    # Generate wiki prompt outputs.
    wiki_prompts = [
        t % e
        for t, s_e in wiki_prompt_splits.items()
        for e in ([s_e['entity']] if s_e['entity']
                  else [a for a in kept_entity_splits
                        if kept_entity_splits[a] == 'train' or s_e['split'] == 'train'])
    ]
    wiki_cache_path = os.path.join(INSTANCE_DIR, f'{INSTANCE}_wiki_prompt_to_output.json')
    if os.path.exists(wiki_cache_path):
        wiki_prompt_to_output = json.load(open(wiki_cache_path))
    else:
        print(f"Generating wiki prompt outputs ({len(wiki_prompts)} prompts)...")
        wiki_outputs = generate_batched(
            model, tokenizer, wiki_prompts,
            max_new_tokens=8, batch_size=64)
        wiki_prompt_to_output = {k: v[len(k):] for k, v in wiki_outputs}
        json.dump(wiki_prompt_to_output, open(wiki_cache_path, 'w'), ensure_ascii=False)

    all_prompt_to_output = {**prompt_to_output, **wiki_prompt_to_output}

    metadata = RAVELMetadata(
        instance=INSTANCE,
        entity_to_split=kept_entity_splits,
        attr_to_prompt=kept_attr_to_prompt_and_split,
        attr_prompt_to_split=kept_prompt_splits,
        entity_prompt_to_split=wiki_prompt_splits,
        prompt_to_output=all_prompt_to_output,
    )

    # Generate splits.
    print("Generating context test split...")
    context_test_data = gen_context_test_split(
        metadata, extract_label_fn=extract_label,
        filter_example_fn=filter_inv_example, first_n=256)
    # Merge subsplits (causal/output/other).
    context_test_merged = collections.defaultdict(list)
    for split in context_test_data:
        context_test_merged[re.sub(r'-causal|-output|-other', '', split)].extend(
            context_test_data[split])
    context_test_data = dict(context_test_merged)

    print("Generating entity test split...")
    entity_test_data = gen_entity_test_split(
        metadata, extract_label_fn=extract_label,
        filter_example_fn=filter_inv_example, first_n=128)
    entity_test_merged = collections.defaultdict(list)
    for split in entity_test_data:
        entity_test_merged[re.sub(r'-causal|-output|-other', '', split)].extend(
            entity_test_data[split])
    entity_test_data = dict(entity_test_merged)

    print("Generating train split...")
    # Use the gen_train_split from the repo (not the Colab's local version).
    from utils.generate_ravel_instance import gen_train_split
    train_data = gen_train_split(
        metadata, extract_label_fn=extract_label,
        filter_example_fn=filter_inv_example, first_n=10240)

    # Save.
    json.dump(train_data, open(train_path, 'w'), ensure_ascii=False)
    json.dump(context_test_data, open(context_test_path, 'w'), ensure_ascii=False)
    json.dump(entity_test_data, open(entity_test_path, 'w'), ensure_ascii=False)

    # Postprocess lat/lon labels.
    for path in [context_test_path, entity_test_path, train_path]:
        data = json.load(open(path))
        for split in data:
            for i in range(len(data[split])):
                if (split.split('-')[0] in ['Latitude', 'Longitude'] or
                        split.split('-')[0] in attribute_prompts.get('Latitude', []) or
                        split.split('-')[0] in attribute_prompts.get('Longitude', [])):
                    data[split][i]['inv_label'] = data[split][i]['inv_label'].replace('\u00b0', '.').split('.')[0]
                    data[split][i]['label'] = data[split][i]['label'].replace('\u00b0', '.').split('.')[0]
        json.dump(data, open(path, 'w'), ensure_ascii=False)

    # Reload after postprocessing.
    split_to_raw = json.load(open(train_path))
    split_to_raw.update(json.load(open(context_test_path)))
    split_to_raw.update(json.load(open(entity_test_path)))
    return split_to_raw, kept_attr_to_prompt_and_split


# ─── Step 5: Compute entity intervention positions ───

def compute_inv_positions(tokenizer, attribute_prompts, wiki_prompt_splits):
    pos_path = os.path.join(INSTANCE_DIR, f'{INSTANCE}_{ENTITY_TYPE}_prompt_to_entity_position.json')
    if os.path.exists(pos_path):
        return json.load(open(pos_path))

    print("Computing entity intervention positions...")
    all_templates = set(wiki_prompt_splits.keys())
    for vs in attribute_prompts.values():
        all_templates.update(vs)

    split_to_inv_position = {}
    for template in all_templates:
        if template.count('%s') != 1:
            continue
        prompt_input = template.replace('%s', '000000', 1)
        input_ids = tokenizer(prompt_input)['input_ids']
        toks = tokenizer.batch_decode(input_ids)
        # Find the position of '000000' tokens — for Llama tokenizer.
        pos = -1
        for i in range(-1, -len(toks), -1):
            if toks[i] == '0' and toks[i - 1] == '0' and toks[i - 2] == '0' and toks[i - 3] == '0':
                pos = i
                break
        split_to_inv_position[template] = pos

    json.dump(split_to_inv_position, open(pos_path, 'w'), ensure_ascii=False, indent=2)
    print(f"Min entity position: {min(split_to_inv_position.values())}")
    return split_to_inv_position


def build_inv_locations(prompt_to_entity_pos):
    split_to_inv_locations = {}
    for task, pos in prompt_to_entity_pos.items():
        for suffix in ('-train', '-test', '-val', ''):
            split_to_inv_locations[f'{task}{suffix}'] = {
                'max_input_length': INPUT_MAX_LEN,
                'inv_position': [INPUT_MAX_LEN + pos],
            }
    return split_to_inv_locations


# ─── Training + eval ───

def run_das(config, model, tokenizer, split_to_dataset):
    from train_intervention import train_intervention
    print(f"\n{'='*60}\nTraining DAS...\n{'='*60}")
    return train_intervention(config, model, tokenizer, split_to_dataset)


def run_complement_das(config, model, tokenizer, split_to_dataset):
    from train_intervention_complement import train_intervention_complement
    print(f"\n{'='*60}\nTraining Complement DAS...\n{'='*60}")
    return train_intervention_complement(config, model, tokenizer, split_to_dataset)


def evaluate(intervenable, split_to_dataset, split_to_inv_locations, tokenizer,
             kept_attr_to_prompt_and_split, method_name):
    print(f"\n{'='*60}\nEvaluating {method_name}...\n{'='*60}")

    eval_split_to_dataset = {
        k: v for k, v in split_to_dataset.items()
        if k.endswith('-test') or k.endswith('-val')
    }

    split_to_eval_metrics = eval_with_interventions(
        intervenable, eval_split_to_dataset, split_to_inv_locations, tokenizer,
        compute_metrics_fn=compute_metrics,
        max_new_tokens=MAX_OUTPUT_TOKENS,
        eval_batch_size=EVAL_BATCH_SIZE)

    eval_path = os.path.join(MODEL_DIR, f'{method_name}_evalall.json')
    json.dump(split_to_eval_metrics, open(eval_path, 'w'))
    print(f"Saved eval to {eval_path}")

    # Disentangle scores on test split.
    split_suffix = '-test'
    attribute_to_iso_tasks = {
        a: [p + split_suffix for p in ps
            if p + split_suffix in eval_split_to_dataset]
        for a, ps in kept_attr_to_prompt_and_split.items() if a != TARGET_ATTR
    }
    attribute_to_cause_tasks = {
        a: [p + split_suffix for p in ps
            if p + split_suffix in eval_split_to_dataset]
        for a, ps in kept_attr_to_prompt_and_split.items() if a == TARGET_ATTR
    }

    scores = compute_disentangle_score(
        split_to_eval_metrics, attribute_to_iso_tasks, attribute_to_cause_tasks)
    print(f"\n--- {method_name} Results ---")
    print(f"  Disentangle: {scores['disentangle']:.4f}")
    print(f"  Isolate:     {scores['isolate']:.4f}")
    print(f"  Cause:       {scores['cause']:.4f}")

    remove_all_forward_hooks(intervenable)
    return scores


# ─── Main ───

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load model.
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print(f"Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")

    # Load base data from data.tgz.
    (entity_attributes, attribute_prompts, prompt_splits,
     entity_splits, wiki_prompt_splits, prompts_to_metadata) = load_base_data()

    # Generate model outputs and filter entities/templates by accuracy.
    prompt_to_output, kept_entity, kept_attr_to_prompt_and_split = (
        generate_outputs_and_filter(
            model, tokenizer, entity_attributes, attribute_prompts,
            prompt_splits, prompts_to_metadata))

    # Create RAVEL instance.
    split_to_raw, kept_attr_to_prompt_and_split = create_ravel_instance(
        model, tokenizer, prompt_to_output,
        kept_entity, kept_attr_to_prompt_and_split,
        entity_splits, wiki_prompt_splits, attribute_prompts)

    # Prepend SOS pad for Llama tokenizer.
    SOS_PAD = '0'
    for split in split_to_raw:
        for i in range(len(split_to_raw[split])):
            split_to_raw[split][i]['inv_label'] = SOS_PAD + split_to_raw[split][i]['inv_label']
            split_to_raw[split][i]['label'] = SOS_PAD + split_to_raw[split][i]['label']

    # Compute intervention positions and build location mapping.
    prompt_to_entity_pos = compute_inv_positions(
        tokenizer, attribute_prompts, wiki_prompt_splits)
    split_to_inv_locations = build_inv_locations(prompt_to_entity_pos)

    # Filter examples and build datasets.
    def filter_example(example):
        return (example['label'] != example['inv_label'] and
                example['source_split'] in split_to_inv_locations and
                example['split'] in split_to_inv_locations)

    for split in list(split_to_raw.keys()):
        random.shuffle(split_to_raw[split])
        split_to_raw[split] = list(filter(filter_example, split_to_raw[split]))
        if len(split_to_raw[split]) == 0:
            print(f'Empty split: "{split}"')
    split_to_raw = {k: v for k, v in split_to_raw.items() if len(v) > 0}

    n_train = sum(len(v) for k, v in split_to_raw.items() if k.endswith('-train'))
    n_test = sum(len(v) for k, v in split_to_raw.items() if k.endswith('-test'))
    n_val = sum(len(v) for k, v in split_to_raw.items() if k.endswith('-val'))
    print(f"#Train={n_train}, #Val={n_val}, #Test={n_test}")

    split_to_dataset = {
        split: Dataset.from_list(split_to_raw[split], features=FEATURE_TYPES)
        for split in split_to_raw
    }

    # ─── Shared config ───
    base_config = {
        'regularization_coefficient': 0,
        'intervention_dimension': INV_DIM,
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'intervenable_config': {
            'intervenable_layer': INV_LAYER,
            'intervenable_representation_type': 'block_output',
            'intervenable_unit': 'pos',
            'max_number_of_units': 1,
            'intervenable_interventions_type': LowRankRotatedSpaceIntervention,
        },
        'training_epoch': TRAINING_EPOCH,
        'split_to_inv_locations': split_to_inv_locations,
        'max_train_percentage': 1.0,
        'init_lr': LR,
        'cause_task_sample_size': CAUSE_TASK_SAMPLE_SIZE,
        'iso_task_sample_size': 5000,
        'training_batch_size': TRAINING_BATCH_SIZE,
        'task_to_prompts': {a: list(ps.keys()) for a, ps in kept_attr_to_prompt_and_split.items()},
    }

    all_results = {}

    # ─── 1. Regular DAS (single-task, cause only) ───
    # Skip if already run — results in DAS_Continent_evalall.json
    das_eval_path = os.path.join(MODEL_DIR, 'DAS_Continent_evalall.json')
    if os.path.exists(das_eval_path):
        print(f"\nSkipping DAS training — eval already exists at {das_eval_path}")
    else:
        das_config = {**base_config}
        das_config['training_tasks'] = {TARGET_ATTR: 'match_source'}
        das_config['log_dir'] = os.path.join(MODEL_DIR, 'logs', 'das_continent')
        os.makedirs(das_config['log_dir'], exist_ok=True)

        das_intervenable, _ = run_das(das_config, model, tokenizer, split_to_dataset)
        torch.save(
            {k: (_get_inv(v)).rotate_layer.weight for k, v in das_intervenable.interventions.items()},
            os.path.join(MODEL_DIR, 'das_continent.pt'))
        all_results['DAS'] = evaluate(
            das_intervenable, split_to_dataset, split_to_inv_locations,
            tokenizer, kept_attr_to_prompt_and_split, "DAS_Continent")
        del das_intervenable
        torch.cuda.empty_cache()

    # ─── 2. Complement DAS ───
    alpha_tag = f"a{COMPLEMENT_LOSS_COEFF:g}"
    comp_config = {**base_config}
    comp_config['training_task'] = TARGET_ATTR
    comp_config['complement_loss_coefficient'] = COMPLEMENT_LOSS_COEFF
    comp_config['log_dir'] = os.path.join(MODEL_DIR, 'logs', f'comp_das_continent_{alpha_tag}')
    os.makedirs(comp_config['log_dir'], exist_ok=True)

    comp_intervenable, _ = run_complement_das(comp_config, model, tokenizer, split_to_dataset)
    torch.save(
        {k: (_get_inv(v)).rotate_layer.weight for k, v in comp_intervenable.interventions.items()},
        os.path.join(MODEL_DIR, f'comp_das_continent_{alpha_tag}.pt'))
    all_results[f'Complement_DAS_{alpha_tag}'] = evaluate(
        comp_intervenable, split_to_dataset, split_to_inv_locations,
        tokenizer, kept_attr_to_prompt_and_split, f"CompDAS_Continent_{alpha_tag}")
    del comp_intervenable
    torch.cuda.empty_cache()

    # ─── Summary ───
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Disentangle':>12} {'Isolate':>12} {'Cause':>12}")
    print("-" * 60)
    for method, scores in all_results.items():
        print(f"{method:<20} {scores['disentangle']:>12.4f} {scores['isolate']:>12.4f} {scores['cause']:>12.4f}")

    json.dump(all_results, open(os.path.join(MODEL_DIR, 'results_summary.json'), 'w'), indent=2)
    print(f"\nResults saved to {os.path.join(MODEL_DIR, 'results_summary.json')}")


if __name__ == '__main__':
    main()
