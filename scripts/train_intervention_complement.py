"""Train DAS with complement-intervention loss (non-multitask).

For every (base, source) pair, runs two forward passes through a shared R:
  1. Standard:   h = b + R^T(Rs - Rb)  → loss against source label
  2. Complement: h = s + R^T(Rb - Rs)  → loss against base label

This gives disentanglement pressure without a separate isolation dataset.
"""

import collections
import numpy as np

from methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
import pyvene as pv
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import get_scheduler
from utils.dataset_utils import get_dataloader
from utils.intervention_utils import (train_intervention_step,
                                      eval_with_interventions,
                                      get_intervention_config,
                                      remove_invalid_token_id)
from utils.metric_utils import compute_metrics, compute_cross_entropy_loss


def train_intervention_complement(config, model, tokenizer, split_to_dataset):
  training_task = config['training_task']
  print('Training task: %s' % training_task)

  # Load single-task dataset.
  train_dataset = split_to_dataset[f'{training_task}-train'].select(
      np.random.choice(
          min(len(split_to_dataset[f'{training_task}-train']),
              config['cause_task_sample_size']),
          size=min(len(split_to_dataset[f'{training_task}-train']),
                   config['cause_task_sample_size']),
          replace=False))
  max_train_example = int(config['max_train_percentage'] * len(train_dataset))
  max_input_length = max([
      v['max_input_length'] for v in config['split_to_inv_locations'].values()
  ])
  train_dataloader = get_dataloader(
      train_dataset.select(range(max_train_example)),
      tokenizer=tokenizer,
      batch_size=config['training_batch_size'],
      prompt_max_length=max_input_length,
      output_max_length=3 + config['max_output_tokens'],
      first_n=config['max_output_tokens'],
      drop_last=True,
      shuffle=True)

  # Create model.
  split_to_inv_locations = config['split_to_inv_locations']
  intervenable_config = get_intervention_config(
      type(model),
      config['intervenable_config']['intervenable_representation_type'],
      config['intervenable_config']['intervenable_layer'],
      LowRankRotatedSpaceIntervention,
      intervention_dimension=config['intervention_dimension'])
  intervenable = pv.IntervenableModel(intervenable_config, model)
  intervenable.set_device(model.device)
  intervenable.disable_model_gradients()

  # Set up optimizer.
  num_epoch = config['training_epoch']
  complement_loss_coefficient = config.get('complement_loss_coefficient', 1.0)
  optimizer_params = []
  for k, v in intervenable.interventions.items():
    optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
  optimizer = torch.optim.AdamW(optimizer_params,
                                lr=config['init_lr'],
                                weight_decay=0)
  scheduler = get_scheduler('constant',
                            optimizer=optimizer,
                            num_training_steps=num_epoch *
                            len(train_dataloader))
  print("Model trainable parameters: ", pv.count_parameters(intervenable.model))
  print("Intervention trainable parameters: ", intervenable.count_parameters())

  # Training loop.
  train_iterator = trange(0, int(num_epoch), desc="Epoch")
  tb_writer = SummaryWriter(config['log_dir'])
  num_output_tokens = config['max_output_tokens']
  for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader,
                          desc=f"Epoch: {epoch}",
                          position=0,
                          leave=True)
    aggregated_stats = collections.defaultdict(list)
    for step, inputs in enumerate(epoch_iterator):
      for k, v in inputs.items():
        if v is not None and isinstance(v, torch.Tensor):
          inputs[k] = v.to(model.device)
      b_s = inputs["input_ids"].shape[0]
      position_ids = {
          f'{prefix}position_ids':
          intervenable.model.prepare_inputs_for_generation(
              input_ids=inputs[f"{prefix}input_ids"],
              attention_mask=inputs[f"{prefix}attention_mask"])['position_ids']
          for prefix in ('', 'source_')
      }
      inputs.update(position_ids)
      for key in inputs:
        if key in ('input_ids', 'source_input_ids', 'attention_mask',
                   'source_attention_mask', 'position_ids',
                   'source_position_ids'):
          inputs[key] = inputs[key].to(model.device)

      # --- Pass 1: standard intervention ---
      for k, v in intervenable.interventions.items():
        v[0].set_complement(False)
      counterfactual_outputs = train_intervention_step(
          intervenable,
          inputs,
          split_to_inv_locations,
          pad_token_id=tokenizer.pad_token_id,
          teacher_label_key='labels')
      loss_cause = compute_cross_entropy_loss(
          counterfactual_outputs.logits,
          inputs["labels"][:, :num_output_tokens],
          next_n_tokens=num_output_tokens,
          pad_token_id=tokenizer.pad_token_id)
      cause_metrics = compute_metrics(
          [counterfactual_outputs.logits[:, :-1]],
          [inputs['labels'][:, :num_output_tokens]],
          last_n_tokens=num_output_tokens,
          pad_token_id=tokenizer.pad_token_id)

      # --- Pass 2: complement intervention ---
      for k, v in intervenable.interventions.items():
        v[0].set_complement(True)
      complement_outputs = train_intervention_step(
          intervenable,
          inputs,
          split_to_inv_locations,
          pad_token_id=tokenizer.pad_token_id,
          teacher_label_key='base_labels')
      loss_iso = compute_cross_entropy_loss(
          complement_outputs.logits,
          inputs["base_labels"][:, :num_output_tokens],
          next_n_tokens=num_output_tokens,
          pad_token_id=tokenizer.pad_token_id)
      iso_metrics = compute_metrics(
          [complement_outputs.logits[:, :-1]],
          [inputs['base_labels'][:, :num_output_tokens]],
          last_n_tokens=num_output_tokens,
          pad_token_id=tokenizer.pad_token_id)

      # Reset complement flag.
      for k, v in intervenable.interventions.items():
        v[0].set_complement(False)

      # Combined loss.
      loss = loss_cause + complement_loss_coefficient * loss_iso

      aggregated_stats['loss'].append(loss.item())
      aggregated_stats['loss_cause'].append(loss_cause.item())
      aggregated_stats['loss_iso'].append(loss_iso.item())
      aggregated_stats['acc_cause'].append(cause_metrics["accuracy"])
      aggregated_stats['acc_iso'].append(iso_metrics["accuracy"])
      epoch_iterator.set_postfix(
          {k: round(np.mean(aggregated_stats[k]), 2) for k in aggregated_stats})

      # Backprop.
      loss.backward()
      optimizer.step()
      scheduler.step()
      intervenable.set_zero_grad()

      # Logging.
      if step % 10 == 0:
        tb_writer.add_scalar("lr",
                             scheduler.get_last_lr()[0], scheduler._step_count)
        tb_writer.add_scalar("loss", loss, scheduler._step_count)
        tb_writer.add_scalar("loss_cause", loss_cause, scheduler._step_count)
        tb_writer.add_scalar("loss_iso", loss_iso, scheduler._step_count)
        tb_writer.add_scalar("acc_cause", cause_metrics["accuracy"],
                             scheduler._step_count)
        tb_writer.add_scalar("acc_iso", iso_metrics["accuracy"],
                             scheduler._step_count)
      if step < 3:
        print('\nTokens to intervene:')
        intervention_locations = [
            split_to_inv_locations[inputs["split"][i]]['inv_position']
            for i in range(len(inputs["split"]))
        ]
        source_intervention_locations = [
            split_to_inv_locations[inputs["source_split"][i]]['inv_position']
            for i in range(len(inputs["split"]))
        ]
        print(inputs['input'][:3])
        print(inputs['source_input'][:3])
        print(
            'Base:',
            tokenizer.batch_decode([
                inputs['input_ids'][i][intervention_locations[i]]
                for i in range(len(inputs["split"]))
            ]))
        print(
            'Source:',
            tokenizer.batch_decode([
                inputs['source_input_ids'][i][source_intervention_locations[i]]
                for i in range(len(inputs["split"]))
            ]))
        print(
            'Cause output:',
            tokenizer.batch_decode(
                torch.argmax(
                    counterfactual_outputs.logits[:, -num_output_tokens - 1:-1],
                    dim=-1)))
        print(
            'Complement output:',
            tokenizer.batch_decode(
                torch.argmax(
                    complement_outputs.logits[:, -num_output_tokens - 1:-1],
                    dim=-1)))
        print(
            'Source label:',
            tokenizer.batch_decode(
                remove_invalid_token_id(inputs['labels'][:, :num_output_tokens],
                                        tokenizer.pad_token_id)))
        print(
            'Base label:',
            tokenizer.batch_decode(
                remove_invalid_token_id(
                    inputs['base_labels'][:, :num_output_tokens],
                    tokenizer.pad_token_id)))
  tb_writer.flush()
  tb_writer.close()
  return intervenable, intervenable_config
