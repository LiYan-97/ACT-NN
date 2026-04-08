
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from actnn_model import BehaviorStructuredACTNN

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DATA_DIR = ROOT_DIR / 'data' / 'model_data'
OUTPUT_ROOT = ROOT_DIR / 'outputs'
EXPERIMENT_REGISTRY = OUTPUT_ROOT / 'experiment_registry.csv'
TOPK_VALUES = (1, 5, 10, 20)
DISTANCE_BINS = [0, 1, 3, 5, 10, 20, 9999]
DISTANCE_BIN_LABELS = ['0-1km', '1-3km', '3-5km', '5-10km', '10-20km', '20km+']
LOSS_TASKS = ['purpose', 'destination', 'depart', 'arrive', 'mode', 'mode_family', 'continue']
GAP_MIN = -2
GAP_MAX = 8
NUM_GAP_CLASSES = GAP_MAX - GAP_MIN + 2
DEFAULT_TIME_BIN_SIZE_HOURS = 2


def build_mode_schema(vocab_mode: dict[str, int], schema: str, old_bos_mode_id: int) -> tuple[dict[str, int], dict[int, int], int]:
    if schema == 'fine7':
        return dict(vocab_mode), {int(v): int(v) for v in vocab_mode.values()}, int(old_bos_mode_id)
    if schema == 'coarse4':
        coarse_vocab = {'[PAD]': 0, 'Active': 1, 'Transit': 2, 'Motorized': 3, 'Other': 4}
        old_to_new = {0: 0}
        old_to_new[vocab_mode.get('Walk', 0)] = 1
        old_to_new[vocab_mode.get('BikeEbike', 0)] = 1
        old_to_new[vocab_mode.get('Bus', 0)] = 2
        old_to_new[vocab_mode.get('Metro', 0)] = 2
        old_to_new[vocab_mode.get('TaxiRidehail', 0)] = 3
        old_to_new[vocab_mode.get('CarMotor', 0)] = 3
        old_to_new[vocab_mode.get('Other', 0)] = 4
        new_bos_mode_id = len(coarse_vocab)
        old_to_new[int(old_bos_mode_id)] = new_bos_mode_id
        return coarse_vocab, old_to_new, new_bos_mode_id
    if schema == 'standard3':
        coarse_vocab = {'[PAD]': 0, 'Active': 1, 'Transit': 2, 'Motorized': 3}
        old_to_new = {0: 0}
        old_to_new[vocab_mode.get('Walk', 0)] = 1
        old_to_new[vocab_mode.get('BikeEbike', 0)] = 1
        old_to_new[vocab_mode.get('Bus', 0)] = 2
        old_to_new[vocab_mode.get('Metro', 0)] = 2
        old_to_new[vocab_mode.get('TaxiRidehail', 0)] = 3
        old_to_new[vocab_mode.get('CarMotor', 0)] = 3
        old_to_new[vocab_mode.get('Other', 0)] = 3
        new_bos_mode_id = len(coarse_vocab)
        old_to_new[int(old_bos_mode_id)] = new_bos_mode_id
        return coarse_vocab, old_to_new, new_bos_mode_id
    raise ValueError(f'Unsupported mode schema: {schema}')


def build_mode_family_lookup(vocab_mode: dict[str, int]) -> np.ndarray:
    lookup = np.zeros(max(vocab_mode.values()) + 1, dtype=np.int32)
    for name, family in {
        'Walk': 0,
        'BikeEbike': 0,
        'Bus': 1,
        'Metro': 1,
        'TaxiRidehail': 2,
        'CarMotor': 2,
        'Other': 3,
    }.items():
        mode_id = int(vocab_mode.get(name, 0))
        if 0 <= mode_id < lookup.shape[0]:
            lookup[mode_id] = family
    return lookup


def remap_mode_arrays(arrays: dict[str, np.ndarray], old_to_new: dict[int, int], new_bos_mode_id: int) -> dict[str, np.ndarray]:
    remapped = {key: value.copy() for key, value in arrays.items()}
    if 'y_mode' in remapped:
        lookup_size = max(max(old_to_new.keys()), int(remapped['y_mode'].max()), int(remapped['step_cat'][:, :, 2].max())) + 1
        lut = np.zeros(lookup_size, dtype=np.int32)
        for old_id, new_id in old_to_new.items():
            if old_id < lookup_size:
                lut[int(old_id)] = int(new_id)
        remapped['y_mode'] = lut[remapped['y_mode'].astype(np.int32)]
        remapped['step_cat'][:, :, 2] = lut[remapped['step_cat'][:, :, 2].astype(np.int32)]
        valid_prev_mode = remapped['seq_mask'] > 0
        remapped['step_cat'][:, :, 2] = np.where(valid_prev_mode & (remapped['step_cat'][:, :, 2] == 0), new_bos_mode_id, remapped['step_cat'][:, :, 2])
        remapped['step_cat'][:, :, 2] = np.where(valid_prev_mode, remapped['step_cat'][:, :, 2], 0)
    return remapped


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def make_run_dir(run_name: str) -> Path:
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figure_data').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    return run_dir


def load_split_arrays(split: str, limit: int | None = None) -> dict[str, np.ndarray]:
    arrays = dict(np.load(MODEL_DATA_DIR / f'{split}_dataset.npz'))
    if limit is not None:
        arrays = {key: value[:limit] for key, value in arrays.items()}
    return arrays

def compute_num_time_classes(time_bin_size_hours: int) -> int:
    return int(math.ceil(24 / float(time_bin_size_hours)))


def aggregate_time_labels(labels: np.ndarray, mask: np.ndarray, time_bin_size_hours: int) -> np.ndarray:
    coarse = labels.copy().astype(np.int32)
    valid = mask > 0
    if time_bin_size_hours <= 1:
        return coarse
    base = np.maximum(coarse.astype(np.int32) - 1, 0)
    coarse_bins = (base // int(time_bin_size_hours)) + 1
    coarse[valid] = coarse_bins[valid]
    coarse[~valid] = 0
    return coarse.astype(np.int32)


def remap_time_arrays(arrays: dict[str, np.ndarray], time_bin_size_hours: int) -> dict[str, np.ndarray]:
    if time_bin_size_hours <= 1:
        return arrays
    remapped = {key: value.copy() for key, value in arrays.items()}
    num_time_classes = compute_num_time_classes(time_bin_size_hours)
    denom = float(max(num_time_classes - 1, 1))

    remapped['y_depart_bin24'] = aggregate_time_labels(remapped['y_depart_bin24'], remapped['mask_depart_bin24'], time_bin_size_hours)
    remapped['y_arrive_bin24'] = aggregate_time_labels(remapped['y_arrive_bin24'], remapped['mask_arrive_bin24'], time_bin_size_hours)

    step_num = remapped['step_num'].astype(np.float32)
    step_num[:, :, 1] = 0.0
    step_num[:, :, 2] = 0.0
    prev_depart = np.maximum(remapped['y_depart_bin24'][:, :-1].astype(np.float32) - 1.0, 0.0)
    prev_arrive = np.maximum(remapped['y_arrive_bin24'][:, :-1].astype(np.float32) - 1.0, 0.0)
    step_num[:, 1:, 1] = prev_depart / denom
    step_num[:, 1:, 2] = prev_arrive / denom
    remapped['step_num'] = step_num
    return remapped


def compute_class_weights(labels: np.ndarray, mask: np.ndarray, num_classes: int, power: float = 0.35) -> np.ndarray:
    valid = labels[mask > 0].astype(np.int64)
    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.sum() > 1:
        total = counts[nonzero].sum()
        base = total / np.maximum(counts[nonzero], 1.0)
        base = np.power(base, power)
        base = base / base.mean()
        weights[nonzero] = base.astype(np.float32)
    weights[0] = 0.0
    return np.clip(weights, 0.0, 4.0)


def build_tf_dataset(arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(arrays)
    if shuffle:
        dataset = dataset.shuffle(min(len(arrays['person_id']), 8192), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_inputs(batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    return {
        'static_cat': batch['static_cat'],
        'static_num': batch['static_num'],
        'step_cat': batch['step_cat'],
        'step_num': batch['step_num'],
        'seq_mask': batch['seq_mask'],
    }


def masked_sparse_ce(logits: tf.Tensor, labels: tf.Tensor, mask: tf.Tensor, class_weights: tf.Tensor | None = None) -> tf.Tensor:
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits)
    if class_weights is not None:
        weights = tf.gather(class_weights, tf.cast(labels, tf.int32))
        losses *= weights
    losses *= tf.cast(mask, tf.float32)
    denom = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(losses) / denom

def masked_huber(y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor, delta: float = 1.5) -> tf.Tensor:
    error = tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    losses = 0.5 * tf.square(quadratic) + delta * linear
    losses *= tf.cast(mask, tf.float32)
    denom = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(losses) / denom


def expected_time_from_logits(logits: tf.Tensor) -> tf.Tensor:
    probs = tf.nn.softmax(logits, axis=-1)
    bins = tf.range(tf.shape(logits)[-1], dtype=tf.float32) - 1.0
    bins = tf.maximum(bins, 0.0)
    return tf.reduce_sum(probs * bins[tf.newaxis, tf.newaxis, :], axis=-1)


def expected_gap_from_logits(logits: tf.Tensor) -> tf.Tensor:
    probs = tf.nn.softmax(logits, axis=-1)
    gap_values = tf.concat([tf.zeros((1,), dtype=tf.float32), tf.range(GAP_MIN, GAP_MAX + 1, dtype=tf.float32)], axis=0)
    return tf.reduce_sum(probs * gap_values[tf.newaxis, tf.newaxis, :], axis=-1)


def derive_gap_targets_from_arrays(y_depart: np.ndarray, y_arrive: np.ndarray, mask_depart: np.ndarray, mask_arrive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gap_labels = np.zeros_like(y_depart, dtype=np.int32)
    gap_mask = np.zeros_like(mask_depart, dtype=np.float32)
    prev_arrive_bin = np.maximum(y_arrive[:, :-1].astype(np.int32) - 1, 0)
    curr_depart_bin = np.maximum(y_depart[:, 1:].astype(np.int32) - 1, 0)
    valid = (mask_depart[:, 1:] > 0) & (mask_arrive[:, :-1] > 0)
    gap_bin = np.clip(curr_depart_bin - prev_arrive_bin, GAP_MIN, GAP_MAX)
    gap_labels[:, 1:] = np.where(valid, gap_bin - GAP_MIN + 1, 0).astype(np.int32)
    gap_mask[:, 1:] = valid.astype(np.float32)
    return gap_labels, gap_mask


def derive_gap_targets_from_batch(batch: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    gap_labels = tf.zeros_like(batch['y_depart_bin24'], dtype=tf.int32)
    gap_mask = tf.zeros_like(batch['mask_depart_bin24'], dtype=tf.float32)
    prev_arrive_bin = tf.maximum(tf.cast(batch['y_arrive_bin24'][:, :-1], tf.int32) - 1, 0)
    curr_depart_bin = tf.maximum(tf.cast(batch['y_depart_bin24'][:, 1:], tf.int32) - 1, 0)
    valid = (batch['mask_depart_bin24'][:, 1:] > 0) & (batch['mask_arrive_bin24'][:, :-1] > 0)
    gap_bin = tf.clip_by_value(curr_depart_bin - prev_arrive_bin, GAP_MIN, GAP_MAX)
    gap_step_labels = tf.where(valid, gap_bin - GAP_MIN + 1, 0)
    gap_labels = tf.concat([gap_labels[:, :1], tf.cast(gap_step_labels, tf.int32)], axis=1)
    gap_mask = tf.concat([gap_mask[:, :1], tf.cast(valid, tf.float32)], axis=1)
    return gap_labels, gap_mask


def derive_duration_targets_from_arrays(
    y_depart: np.ndarray,
    y_arrive: np.ndarray,
    mask_depart: np.ndarray,
    mask_arrive: np.ndarray,
    num_time_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    depart_bin = np.maximum(y_depart.astype(np.int32) - 1, 0)
    arrive_bin = np.maximum(y_arrive.astype(np.int32) - 1, 0)
    duration_bin = np.clip(arrive_bin - depart_bin, 0, num_time_classes - 1)
    duration_label = np.where((mask_depart > 0) & (mask_arrive > 0), duration_bin + 1, 0).astype(np.int32)
    duration_mask = ((mask_depart > 0) & (mask_arrive > 0)).astype(np.float32)
    return duration_label, duration_mask


def derive_duration_targets_from_batch(batch: dict[str, tf.Tensor], num_time_classes: int) -> tuple[tf.Tensor, tf.Tensor]:
    depart_bin = tf.maximum(tf.cast(batch['y_depart_bin24'], tf.int32) - 1, 0)
    arrive_bin = tf.maximum(tf.cast(batch['y_arrive_bin24'], tf.int32) - 1, 0)
    duration_bin = tf.clip_by_value(arrive_bin - depart_bin, 0, num_time_classes - 1)
    duration_mask = tf.cast((batch['mask_depart_bin24'] > 0) & (batch['mask_arrive_bin24'] > 0), tf.float32)
    duration_label = tf.where(duration_mask > 0, duration_bin + 1, 0)
    return tf.cast(duration_label, tf.int32), duration_mask

def compute_step_log_prior_matrix(
    labels: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    smoothing: float = 1.0,
    has_pad_zero: bool = True,
) -> np.ndarray:
    priors = np.zeros((labels.shape[1], num_classes), dtype=np.float32)
    for step_idx in range(labels.shape[1]):
        valid = labels[:, step_idx][mask[:, step_idx] > 0].astype(np.int64)
        counts = np.full(num_classes, smoothing, dtype=np.float64)
        if has_pad_zero:
            counts[0] = 0.0
        if valid.size > 0:
            counts += np.bincount(valid, minlength=num_classes)
            if has_pad_zero:
                counts[0] = 0.0
            probs = counts / np.maximum(counts.sum(), 1.0)
            log_probs = np.log(np.clip(probs, 1e-8, 1.0))
            if has_pad_zero and num_classes > 1:
                log_probs[1:] -= log_probs[1:].mean()
            elif num_classes > 1:
                log_probs -= log_probs.mean()
            priors[step_idx] = log_probs.astype(np.float32)
    return priors


def compute_purpose_mode_log_prior(
    purpose_labels: np.ndarray,
    purpose_mask: np.ndarray,
    mode_labels: np.ndarray,
    mode_mask: np.ndarray,
    num_purpose: int,
    num_mode: int,
    smoothing: float = 1.0,
) -> np.ndarray:
    counts = np.full((num_purpose, num_mode), smoothing, dtype=np.float64)
    counts[:, 0] = 0.0
    valid = (purpose_mask > 0) & (mode_mask > 0)
    flat_purpose = purpose_labels[valid].astype(np.int64)
    flat_mode = mode_labels[valid].astype(np.int64)
    for p, m in zip(flat_purpose, flat_mode):
        if 0 < p < num_purpose and 0 < m < num_mode:
            counts[p, m] += 1.0
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    if num_mode > 1:
        log_probs[:, 1:] -= log_probs[:, 1:].mean(axis=1, keepdims=True)
    log_probs[0, :] = 0.0
    return log_probs.astype(np.float32)




def compute_conditional_log_prior(
    cond_labels: np.ndarray,
    cond_mask: np.ndarray,
    target_labels: np.ndarray,
    target_mask: np.ndarray,
    num_cond: int,
    num_target: int,
    smoothing: float = 1.0,
) -> np.ndarray:
    counts = np.full((num_cond, num_target), smoothing, dtype=np.float64)
    counts[:, 0] = 0.0
    valid = (cond_mask > 0) & (target_mask > 0)
    cond_flat = cond_labels[valid].astype(np.int64)
    target_flat = target_labels[valid].astype(np.int64)
    for c, t in zip(cond_flat, target_flat):
        if 0 < c < num_cond and 0 < t < num_target:
            counts[c, t] += 1.0
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    if num_target > 1:
        log_probs[:, 1:] -= log_probs[:, 1:].mean(axis=1, keepdims=True)
    log_probs[0, :] = 0.0
    return log_probs.astype(np.float32)


def prepare_time_behavior_priors(train_arrays: dict[str, np.ndarray], num_purpose: int, num_mode: int, num_time_classes: int) -> dict[str, np.ndarray]:
    duration_labels, duration_mask = derive_duration_targets_from_arrays(
        train_arrays['y_depart_bin24'],
        train_arrays['y_arrive_bin24'],
        train_arrays['mask_depart_bin24'],
        train_arrays['mask_arrive_bin24'],
        num_time_classes,
    )
    return {
        'purpose_depart': compute_conditional_log_prior(
            train_arrays['y_purpose'],
            train_arrays['mask_purpose'],
            train_arrays['y_depart_bin24'],
            train_arrays['mask_depart_bin24'],
            num_purpose,
            num_time_classes + 1,
        ),
        'purpose_duration': compute_conditional_log_prior(
            train_arrays['y_purpose'],
            train_arrays['mask_purpose'],
            duration_labels,
            duration_mask,
            num_purpose,
            num_time_classes + 1,
        ),
        'mode_duration': compute_conditional_log_prior(
            train_arrays['y_mode'],
            train_arrays['mask_mode'],
            duration_labels,
            duration_mask,
            num_mode,
            num_time_classes + 1,
        ),
    }
def prepare_purpose_step_priors(train_arrays: dict[str, np.ndarray], num_purpose: int) -> np.ndarray:
    return compute_step_log_prior_matrix(train_arrays['y_purpose'], train_arrays['mask_purpose'], num_purpose)


def prepare_home_return_step_priors(train_arrays: dict[str, np.ndarray]) -> np.ndarray:
    max_steps = train_arrays['y_dest_zone'].shape[1]
    priors = np.zeros((max_steps,), dtype=np.float32)
    home_zone = train_arrays['static_cat'][:, 5][:, None]
    dest_is_home = (train_arrays['y_dest_zone'] == home_zone) & (train_arrays['mask_dest_zone'] > 0)
    for step_idx in range(max_steps):
        valid = train_arrays['mask_dest_zone'][:, step_idx] > 0
        if not np.any(valid):
            continue
        p = float(dest_is_home[valid, step_idx].mean())
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        priors[step_idx] = float(np.log(p / (1.0 - p)))
    priors -= priors.mean()
    return priors.astype(np.float32)
def prepare_mode_priors(train_arrays: dict[str, np.ndarray], num_purpose: int, num_mode: int) -> dict[str, np.ndarray]:
    return {
        'step': compute_step_log_prior_matrix(train_arrays['y_mode'], train_arrays['mask_mode'], num_mode),
        'purpose': compute_purpose_mode_log_prior(
            train_arrays['y_purpose'],
            train_arrays['mask_purpose'],
            train_arrays['y_mode'],
            train_arrays['mask_mode'],
            num_purpose,
            num_mode,
        ),
    }


def prepare_mode_transition_priors(train_arrays: dict[str, np.ndarray], num_prev_mode: int, num_mode: int) -> np.ndarray:
    counts = np.full((num_prev_mode, num_mode), 1.0, dtype=np.float64)
    counts[:, 0] = 0.0
    prev_mode = train_arrays['step_cat'][:, :, 2].astype(np.int32)
    curr_mode = train_arrays['y_mode'].astype(np.int32)
    valid = (train_arrays['mask_mode'] > 0) & (prev_mode >= 0) & (prev_mode < num_prev_mode)
    if not np.any(valid):
        return np.zeros((num_prev_mode, num_mode), dtype=np.float32)
    for pm, cm in zip(prev_mode[valid], curr_mode[valid]):
        if 0 <= pm < num_prev_mode and 0 < cm < num_mode:
            counts[int(pm), int(cm)] += 1.0
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    log_probs -= log_probs.mean(axis=1, keepdims=True)
    return log_probs.astype(np.float32)


def prepare_mode_usual_commute_priors(train_arrays: dict[str, np.ndarray], num_usual_commute: int, num_mode: int) -> np.ndarray:
    counts = np.full((num_usual_commute, num_mode), 1.0, dtype=np.float64)
    counts[:, 0] = 0.0
    usual_commute = train_arrays['static_cat'][:, 4].astype(np.int32)
    curr_mode = train_arrays['y_mode'].astype(np.int32)
    valid = train_arrays['mask_mode'] > 0
    if not np.any(valid):
        return np.zeros((num_usual_commute, num_mode), dtype=np.float32)
    tiled_commute = np.repeat(usual_commute[:, None], curr_mode.shape[1], axis=1)
    for uc, cm in zip(tiled_commute[valid], curr_mode[valid]):
        if 0 <= uc < num_usual_commute and 0 < cm < num_mode:
            counts[int(uc), int(cm)] += 1.0
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    log_probs -= log_probs.mean(axis=1, keepdims=True)
    return log_probs.astype(np.float32)


def prepare_mode_distance_priors(train_arrays: dict[str, np.ndarray], zone_coords: np.ndarray, num_mode: int) -> np.ndarray:
    counts = np.full((len(DISTANCE_BIN_LABELS), num_mode), 1.0, dtype=np.float64)
    counts[:, 0] = 0.0
    origin_idx = train_arrays['step_cat'][:, :, 0].astype(np.int32)
    dest_idx = train_arrays['y_dest_zone'].astype(np.int32)
    mode = train_arrays['y_mode'].astype(np.int32)
    valid = (train_arrays['mask_mode'] > 0) & (train_arrays['mask_dest_zone'] > 0) & (origin_idx > 0) & (dest_idx > 0)
    if not np.any(valid):
        return np.zeros((len(DISTANCE_BIN_LABELS), num_mode), dtype=np.float32)
    origin_lon = zone_coords[origin_idx[valid], 0]
    origin_lat = zone_coords[origin_idx[valid], 1]
    dest_lon = zone_coords[dest_idx[valid], 0]
    dest_lat = zone_coords[dest_idx[valid], 1]
    dist = haversine_km_np(origin_lon, origin_lat, dest_lon, dest_lat)
    bin_ids = np.digitize(dist, DISTANCE_BINS[1:-1], right=False)
    mode_vals = mode[valid]
    for b, m in zip(bin_ids, mode_vals):
        if 0 <= b < len(DISTANCE_BIN_LABELS) and 0 < m < num_mode:
            counts[int(b), int(m)] += 1.0
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    if num_mode > 1:
        log_probs[:, 1:] -= log_probs[:, 1:].mean(axis=1, keepdims=True)
    return log_probs.astype(np.float32)


def prepare_step_time_priors(train_arrays: dict[str, np.ndarray], num_time_classes: int) -> dict[str, np.ndarray]:
    gap_labels, gap_mask = derive_gap_targets_from_arrays(
        train_arrays['y_depart_bin24'],
        train_arrays['y_arrive_bin24'],
        train_arrays['mask_depart_bin24'],
        train_arrays['mask_arrive_bin24'],
    )
    duration_labels, duration_mask = derive_duration_targets_from_arrays(
        train_arrays['y_depart_bin24'],
        train_arrays['y_arrive_bin24'],
        train_arrays['mask_depart_bin24'],
        train_arrays['mask_arrive_bin24'],
        num_time_classes,
    )
    return {
        'first_depart': compute_step_log_prior_matrix(train_arrays['y_depart_bin24'], train_arrays['mask_depart_bin24'], num_time_classes + 1),
        'gap': compute_step_log_prior_matrix(gap_labels, gap_mask, NUM_GAP_CLASSES),
        'duration': compute_step_log_prior_matrix(duration_labels, duration_mask, num_time_classes + 1),
        'continue': compute_step_log_prior_matrix(train_arrays['y_continue'], train_arrays['mask_continue'], 2, has_pad_zero=False),
    }



def prepare_continue_home_priors(train_arrays: dict[str, np.ndarray]) -> np.ndarray:
    max_steps = train_arrays['y_continue'].shape[1]
    priors = np.zeros((max_steps, 2, 2), dtype=np.float32)
    home_zone = train_arrays['static_cat'][:, 5][:, None]
    dest_is_home = (train_arrays['y_dest_zone'] == home_zone) & (train_arrays['mask_dest_zone'] > 0)
    for step_idx in range(max_steps):
        valid_step = train_arrays['mask_continue'][:, step_idx] > 0
        if not np.any(valid_step):
            continue
        for home_state in (0, 1):
            valid = valid_step & (dest_is_home[:, step_idx] == bool(home_state))
            counts = np.full(2, 1.0, dtype=np.float64)
            if np.any(valid):
                counts += np.bincount(train_arrays['y_continue'][valid, step_idx].astype(np.int64), minlength=2)
            probs = counts / np.maximum(counts.sum(), 1.0)
            log_probs = np.log(np.clip(probs, 1e-8, 1.0))
            log_probs -= log_probs.mean()
            priors[step_idx, home_state, :] = log_probs.astype(np.float32)
    return priors
def derive_depart_labels_from_prev_arrive_and_gap(prev_arrive_labels: np.ndarray, gap_labels: np.ndarray, num_time_classes: int) -> np.ndarray:
    prev_arrive = np.asarray(prev_arrive_labels, dtype=np.int32)
    gap = np.asarray(gap_labels, dtype=np.int32)
    prev_arrive_bin = np.maximum(prev_arrive - 1, 0)
    gap_bin = np.where(gap > 0, gap + GAP_MIN - 1, 0)
    depart_bin = np.clip(prev_arrive_bin + gap_bin, 0, num_time_classes - 1)
    return (depart_bin + 1).astype(np.int32)


def derive_arrive_labels_from_depart_duration(depart_labels: np.ndarray, duration_labels: np.ndarray, num_time_classes: int) -> np.ndarray:
    depart_bin = np.maximum(np.asarray(depart_labels, dtype=np.int32) - 1, 0)
    duration_bin = np.maximum(np.asarray(duration_labels, dtype=np.int32) - 1, 0)
    arrive_bin = np.clip(depart_bin + duration_bin, 0, num_time_classes - 1)
    return (arrive_bin + 1).astype(np.int32)


def decode_time_sequences_from_heads(
    first_depart_labels: np.ndarray,
    gap_labels: np.ndarray,
    duration_labels: np.ndarray,
    seq_mask: np.ndarray,
    num_time_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    first_depart_labels = np.asarray(first_depart_labels, dtype=np.int32)
    gap_labels = np.asarray(gap_labels, dtype=np.int32)
    duration_labels = np.asarray(duration_labels, dtype=np.int32)
    valid = np.asarray(seq_mask) > 0
    depart = np.zeros_like(first_depart_labels, dtype=np.int32)
    arrive = np.zeros_like(first_depart_labels, dtype=np.int32)
    if first_depart_labels.shape[1] == 0:
        return depart, arrive
    first_active = valid[:, 0]
    if np.any(first_active):
        depart[first_active, 0] = first_depart_labels[first_active, 0]
        first_arrive = derive_arrive_labels_from_depart_duration(depart[:, :1], duration_labels[:, :1], num_time_classes)[:, 0]
        arrive[first_active, 0] = first_arrive[first_active]
    for step_idx in range(1, first_depart_labels.shape[1]):
        active = valid[:, step_idx]
        if not np.any(active):
            continue
        depart_step = derive_depart_labels_from_prev_arrive_and_gap(arrive[:, step_idx - 1], gap_labels[:, step_idx], num_time_classes)
        depart[active, step_idx] = depart_step[active]
        arrive_step = derive_arrive_labels_from_depart_duration(
            depart[:, step_idx : step_idx + 1], duration_labels[:, step_idx : step_idx + 1], num_time_classes
        )[:, 0]
        arrive[active, step_idx] = arrive_step[active]
    return depart, arrive


def expected_depart_arrive_from_outputs(outputs: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    first_depart_expect = expected_time_from_logits(outputs['first_depart_logits'])
    gap_expect = expected_gap_from_logits(outputs['gap_logits'])
    duration_expect = expected_time_from_logits(outputs['duration_logits'])

    first_depart_steps = tf.unstack(first_depart_expect, axis=1)
    gap_steps = tf.unstack(gap_expect, axis=1)
    duration_steps = tf.unstack(duration_expect, axis=1)

    depart_steps = []
    arrive_steps = []
    prev_arrive = None
    max_bin = tf.cast(tf.shape(outputs['first_depart_logits'])[-1] - 2, tf.float32)
    for step_idx, duration_step in enumerate(duration_steps):
        if step_idx == 0:
            depart_step = first_depart_steps[0]
        else:
            depart_step = tf.clip_by_value(prev_arrive + gap_steps[step_idx], 0.0, max_bin)
        arrive_step = tf.clip_by_value(depart_step + duration_step, 0.0, max_bin)
        depart_steps.append(depart_step)
        arrive_steps.append(arrive_step)
        prev_arrive = arrive_step

    return tf.stack(depart_steps, axis=1), tf.stack(arrive_steps, axis=1)


def temporal_consistency_penalty(outputs: dict[str, tf.Tensor], seq_mask: tf.Tensor) -> tf.Tensor:
    depart_expect, arrive_expect = expected_depart_arrive_from_outputs(outputs)
    pair_mask = seq_mask[:, 1:] * seq_mask[:, :-1]
    chain_penalty = tf.nn.relu(arrive_expect[:, :-1] - depart_expect[:, 1:] - 1.0)
    return tf.reduce_sum(chain_penalty * pair_mask) / (tf.reduce_sum(pair_mask) + 1e-6)

def compute_total_loss(
    model: BehaviorStructuredACTNN,
    batch: dict[str, tf.Tensor],
    class_weights: dict[str, tf.Tensor],
    loss_weights: dict[str, float],
    temporal_lambda: float,
    time_reg_lambda: float,
    training: bool,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    outputs = model(split_inputs(batch), training=training)
    gap_labels, gap_mask = derive_gap_targets_from_batch(batch)
    duration_labels, duration_mask = derive_duration_targets_from_batch(batch, model.num_time - 1)
    depart_expect, arrive_expect = expected_depart_arrive_from_outputs(outputs)
    depart_true = tf.maximum(tf.cast(batch['y_depart_bin24'], tf.float32) - 1.0, 0.0)
    arrive_true = tf.maximum(tf.cast(batch['y_arrive_bin24'], tf.float32) - 1.0, 0.0)

    first_depart_loss = masked_sparse_ce(
        outputs['first_depart_logits'][:, :1, :],
        batch['y_depart_bin24'][:, :1],
        batch['mask_depart_bin24'][:, :1],
        class_weights['depart_first'],
    )
    gap_loss = masked_sparse_ce(
        outputs['gap_logits'][:, 1:, :],
        gap_labels[:, 1:],
        gap_mask[:, 1:],
        class_weights['depart_gap'],
    )
    first_count = tf.reduce_sum(batch['mask_depart_bin24'][:, :1])
    gap_count = tf.reduce_sum(gap_mask[:, 1:])
    depart_loss = (first_depart_loss * first_count + gap_loss * gap_count) / (first_count + gap_count + 1e-6)

    depart_reg = masked_huber(depart_true, depart_expect, batch['mask_depart_bin24'])
    arrive_reg = masked_huber(arrive_true, arrive_expect, batch['mask_arrive_bin24'])

    mode_family_lookup = tf.argmax(tf.transpose(model.mode_family_membership), axis=-1, output_type=tf.int32)
    mode_family_labels = tf.gather(mode_family_lookup, tf.cast(batch['y_mode'], tf.int32))

    losses = {
        'purpose': masked_sparse_ce(outputs['purpose_logits'], batch['y_purpose'], batch['mask_purpose'], class_weights['purpose']),
        'destination': masked_sparse_ce(outputs['destination_logits'], batch['y_dest_zone'], batch['mask_dest_zone'], None),
        'depart': depart_loss + tf.cast(time_reg_lambda, tf.float32) * depart_reg,
        'arrive': masked_sparse_ce(outputs['duration_logits'], duration_labels, duration_mask, class_weights['arrive']) + tf.cast(time_reg_lambda, tf.float32) * arrive_reg,
        'mode': masked_sparse_ce(outputs['mode_logits'], batch['y_mode'], batch['mask_mode'], class_weights['mode']),
        'mode_family': masked_sparse_ce(outputs['mode_family_logits'], mode_family_labels, batch['mask_mode'], class_weights['mode_family']),
        'continue': masked_sparse_ce(outputs['continue_logits'], batch['y_continue'], batch['mask_continue'], class_weights['continue']),
    }
    losses['depart_reg'] = depart_reg
    losses['arrive_reg'] = arrive_reg
    temporal_penalty = temporal_consistency_penalty(outputs, tf.cast(batch['seq_mask'], tf.float32))
    total = tf.add_n([tf.cast(loss_weights[name], tf.float32) * losses[name] for name in LOSS_TASKS])
    total += tf.cast(temporal_lambda, tf.float32) * temporal_penalty
    losses['temporal_penalty'] = temporal_penalty
    losses['total'] = total
    return total, losses, outputs


@tf.function
def train_step(
    model: BehaviorStructuredACTNN,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: dict[str, tf.Tensor],
    class_weights: dict[str, tf.Tensor],
    loss_weights: dict[str, float],
    temporal_lambda: float,
    time_reg_lambda: float,
) -> dict[str, tf.Tensor]:
    with tf.GradientTape() as tape:
        total_loss, loss_parts, _ = compute_total_loss(model, batch, class_weights, loss_weights, temporal_lambda, time_reg_lambda, training=True)
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_parts


def collect_loss_history_item(loss_sums: dict[str, float], loss_parts: dict[str, tf.Tensor]) -> None:
    for key, value in loss_parts.items():
        loss_sums[key] = loss_sums.get(key, 0.0) + float(value.numpy())


def decode_metrics_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    if y_true.size == 0:
        return float('nan')
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def masked_accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if valid.sum() == 0:
        return float('nan')
    return float((y_true[valid] == y_pred[valid]).mean())


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if valid.sum() == 0:
        return float('nan')
    return float(np.abs(y_true[valid] - y_pred[valid]).mean())


def topk_accuracy(hits: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if valid.sum() == 0:
        return float('nan')
    return float(hits[valid].mean())


def haversine_km_np(origin_lon, origin_lat, dest_lon, dest_lat):
    lon1 = np.radians(origin_lon)
    lat1 = np.radians(origin_lat)
    lon2 = np.radians(dest_lon)
    lat2 = np.radians(dest_lat)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0088 * c


def prepare_schedule_sampling_batch(
    model: BehaviorStructuredACTNN,
    batch: dict[str, tf.Tensor],
    ss_prob: float,
) -> dict[str, tf.Tensor]:
    if ss_prob <= 1e-8:
        return batch

    batch_np = {key: value.numpy().copy() for key, value in batch.items()}
    teacher_inputs = {
        'static_cat': tf.convert_to_tensor(batch_np['static_cat']),
        'static_num': tf.convert_to_tensor(batch_np['static_num']),
        'step_cat': tf.convert_to_tensor(batch_np['step_cat']),
        'step_num': tf.convert_to_tensor(batch_np['step_num']),
        'seq_mask': tf.convert_to_tensor(batch_np['seq_mask']),
    }
    outputs = model(teacher_inputs, training=False, return_attention=False)
    pred_purpose = tf.argmax(outputs['purpose_logits'], axis=-1, output_type=tf.int32).numpy()
    pred_dest = tf.argmax(outputs['destination_logits'], axis=-1, output_type=tf.int32).numpy()
    pred_first_depart = tf.argmax(outputs['first_depart_logits'], axis=-1, output_type=tf.int32).numpy()
    pred_gap = tf.argmax(outputs['gap_logits'], axis=-1, output_type=tf.int32).numpy()
    pred_duration = tf.argmax(outputs['duration_logits'], axis=-1, output_type=tf.int32).numpy()
    pred_depart, pred_arrive = decode_time_sequences_from_heads(
        pred_first_depart, pred_gap, pred_duration, batch_np['seq_mask'], model.num_time - 1
    )
    pred_mode = tf.argmax(outputs['mode_logits'], axis=-1, output_type=tf.int32).numpy()

    seq_mask = batch_np['seq_mask'] > 0
    static_cat = batch_np['static_cat']
    batch_size, max_steps, _ = batch_np['step_cat'].shape
    use_pred_matrix = np.random.rand(batch_size, max_steps - 1) < ss_prob

    for step_idx in range(max_steps - 1):
        use_pred = use_pred_matrix[:, step_idx] & seq_mask[:, step_idx + 1]
        if not np.any(use_pred):
            continue
        batch_np['step_cat'][use_pred, step_idx + 1, 0] = pred_dest[use_pred, step_idx]
        batch_np['step_cat'][use_pred, step_idx + 1, 1] = pred_purpose[use_pred, step_idx]
        batch_np['step_cat'][use_pred, step_idx + 1, 2] = pred_mode[use_pred, step_idx]
        batch_np['step_cat'][use_pred, step_idx + 1, 3] = 0
        time_denom = float(max(model.num_time - 2, 1))
        batch_np['step_num'][use_pred, step_idx + 1, 1] = np.maximum(pred_depart[use_pred, step_idx] - 1, 0) / time_denom
        batch_np['step_num'][use_pred, step_idx + 1, 2] = np.maximum(pred_arrive[use_pred, step_idx] - 1, 0) / time_denom
        batch_np['step_num'][use_pred, step_idx + 1, 3] = (pred_dest[use_pred, step_idx] == static_cat[use_pred, 4]).astype(np.float32)

    return {key: tf.convert_to_tensor(value) for key, value in batch_np.items()}


def compute_schedule_sampling_prob(epoch: int, total_epochs: int, warmup_epochs: int, max_prob: float) -> float:
    if max_prob <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    denom = max(total_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / denom
    return float(max_prob * min(max(progress, 0.0), 1.0))

def prepare_zone_support() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    zone_table = pd.read_csv(MODEL_DATA_DIR / 'zone_feature_table.csv')
    zone_table = zone_table.sort_values('zone_idx').reset_index(drop=True)
    num_zones = int(zone_table['zone_idx'].max()) + 1

    coords = np.zeros((num_zones, 2), dtype=np.float32)
    coords[zone_table['zone_idx'].to_numpy(dtype=np.int32), 0] = zone_table['centroid_lon'].to_numpy(dtype=np.float32)
    coords[zone_table['zone_idx'].to_numpy(dtype=np.int32), 1] = zone_table['centroid_lat'].to_numpy(dtype=np.float32)

    purpose_raw = np.zeros((num_zones, 7), dtype=np.float32)
    purpose_columns = ['Home', 'Work', 'School', 'Shopping', 'Visit', 'OtherGoal']
    for col in purpose_columns:
        if col not in zone_table.columns:
            zone_table[col] = 0.0
    for zone_idx, row in zone_table.set_index('zone_idx')[purpose_columns].iterrows():
        purpose_raw[int(zone_idx), 1:] = row.to_numpy(dtype=np.float32)
    support_binary = (purpose_raw > 0).astype(np.float32)
    row_sum = purpose_raw.sum(axis=1, keepdims=True)
    normalized = np.divide(purpose_raw, np.where(row_sum > 0, row_sum, 1.0), where=row_sum >= 0)
    zone_idx_to_id = dict(zip(zone_table['zone_idx'].astype(int), zone_table['zone_id'].astype(int)))
    return normalized.astype(np.float32), support_binary.astype(np.float32), coords, zone_idx_to_id


def forward_batches(
    model: BehaviorStructuredACTNN,
    arrays: dict[str, np.ndarray],
    batch_size: int,
    home_purpose_id: int,
    zone_support_binary: np.ndarray,
    zone_coords: np.ndarray,
    case_person_limit: int = 30,
    autoregressive: bool = False,
) -> dict[str, Any]:
    n = len(arrays['person_id'])
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in ['purpose', 'destination', 'depart', 'arrive', 'mode', 'continue']}
    dest_hits: dict[int, list[np.ndarray]] = {k: [] for k in TOPK_VALUES}
    used_origin_history: list[np.ndarray] = []
    case_rows: list[dict[str, Any]] = []
    destination_topk_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    captured_case_people: set[int] = set()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_np = {key: value[start:end] for key, value in arrays.items()}
        batch_persons = batch_np['person_id']
        need_attention = len(captured_case_people) < min(case_person_limit, 5)
        used_origin_idx = batch_np['step_cat'][:, :, 0].copy()

        if not autoregressive:
            batch = {key: tf.convert_to_tensor(value) for key, value in batch_np.items()}
            outputs = model(split_inputs(batch), training=False, return_attention=need_attention)
            purpose_pred = tf.argmax(outputs['purpose_logits'], axis=-1, output_type=tf.int32).numpy()
            dest_pred = tf.argmax(outputs['destination_logits'], axis=-1, output_type=tf.int32).numpy()
            first_depart_pred = tf.argmax(outputs['first_depart_logits'], axis=-1, output_type=tf.int32).numpy()
            gap_pred = tf.argmax(outputs['gap_logits'], axis=-1, output_type=tf.int32).numpy()
            duration_pred = tf.argmax(outputs['duration_logits'], axis=-1, output_type=tf.int32).numpy()
            depart_pred, arrive_pred = decode_time_sequences_from_heads(
                first_depart_pred, gap_pred, duration_pred, batch_np['seq_mask'], model.num_time - 1
            )
            mode_pred = tf.argmax(outputs['mode_logits'], axis=-1, output_type=tf.int32).numpy()
            continue_pred = tf.argmax(outputs['continue_logits'], axis=-1, output_type=tf.int32).numpy()
            topk_indices = tf.math.top_k(outputs['destination_logits'], k=max(TOPK_VALUES)).indices.numpy()
            final_attention_outputs = outputs
        else:
            step_cat_ar = batch_np['step_cat'].copy()
            infer_seq_mask = np.zeros_like(batch_np['seq_mask'])
            step_num_ar = batch_np['step_num'].copy()
            step_cat_ar[:, :, 3] = 0
            batch_size_local, max_steps, _ = step_cat_ar.shape
            home_zone_idx_batch = batch_np['static_cat'][:, 5].copy()
            current_origin = home_zone_idx_batch.copy()
            purpose_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            dest_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            depart_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            arrive_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            mode_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            continue_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            topk_indices = np.zeros((batch_size_local, max_steps, max(TOPK_VALUES)), dtype=np.int32)
            active_mask = np.ones(batch_size_local, dtype=bool)
            continue_home_majority = np.argmax(model.continue_home_prior_matrix.numpy(), axis=-1)

            for step_idx in range(max_steps):
                if not np.any(active_mask):
                    break
                infer_seq_mask[active_mask, step_idx] = 1.0
                step_cat_ar[active_mask, step_idx, 0] = current_origin[active_mask]
                used_origin_idx[active_mask, step_idx] = current_origin[active_mask]
                step_num_ar[active_mask, step_idx, 3] = (current_origin[active_mask] == home_zone_idx_batch[active_mask]).astype(np.float32)
                if step_idx == 0:
                    step_cat_ar[active_mask, step_idx, 1] = model.bos_purpose_id
                    step_cat_ar[active_mask, step_idx, 2] = model.bos_mode_id
                    step_num_ar[active_mask, step_idx, 1] = 0.0
                    step_num_ar[active_mask, step_idx, 2] = 0.0

                batch_inputs = {
                    'static_cat': tf.convert_to_tensor(batch_np['static_cat']),
                    'static_num': tf.convert_to_tensor(batch_np['static_num']),
                    'step_cat': tf.convert_to_tensor(step_cat_ar),
                    'step_num': tf.convert_to_tensor(step_num_ar),
                    'seq_mask': tf.convert_to_tensor(infer_seq_mask),
                }
                outputs_step = model(batch_inputs, training=False, return_attention=False)
                purpose_step = tf.argmax(outputs_step['purpose_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                dest_step = tf.argmax(outputs_step['destination_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                first_depart_step = tf.argmax(outputs_step['first_depart_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                gap_step = tf.argmax(outputs_step['gap_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                duration_step = tf.argmax(outputs_step['duration_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                if step_idx == 0:
                    depart_step = first_depart_step
                else:
                    depart_step = derive_depart_labels_from_prev_arrive_and_gap(arrive_pred[:, step_idx - 1], gap_step, model.num_time - 1)
                arrive_step = derive_arrive_labels_from_depart_duration(depart_step, duration_step, model.num_time - 1)
                mode_step = tf.argmax(outputs_step['mode_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                continue_step = tf.argmax(outputs_step['continue_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                topk_step = tf.math.top_k(outputs_step['destination_logits'][:, step_idx, :], k=max(TOPK_VALUES)).indices.numpy()

                purpose_pred[:, step_idx] = purpose_step
                dest_pred[:, step_idx] = dest_step
                depart_pred[:, step_idx] = depart_step
                arrive_pred[:, step_idx] = arrive_step
                mode_pred[:, step_idx] = mode_step
                continue_pred[:, step_idx] = continue_step
                topk_indices[:, step_idx, :] = topk_step

                next_idx = step_idx + 1
                if next_idx < max_steps:
                    next_active = active_mask & (continue_step == 1)
                    if np.any(next_active):
                        current_origin[next_active] = dest_step[next_active]
                        step_cat_ar[next_active, next_idx, 1] = purpose_step[next_active]
                        step_cat_ar[next_active, next_idx, 2] = mode_step[next_active]
                        time_denom = float(max(model.num_time - 2, 1))
                        step_num_ar[next_active, next_idx, 1] = np.maximum(depart_step[next_active] - 1, 0) / time_denom
                        step_num_ar[next_active, next_idx, 2] = np.maximum(arrive_step[next_active] - 1, 0) / time_denom
                    active_mask = next_active
                else:
                    active_mask = np.zeros(batch_size_local, dtype=bool)

            final_inputs = {
                'static_cat': tf.convert_to_tensor(batch_np['static_cat']),
                'static_num': tf.convert_to_tensor(batch_np['static_num']),
                'step_cat': tf.convert_to_tensor(step_cat_ar),
                'step_num': tf.convert_to_tensor(step_num_ar),
                'seq_mask': tf.convert_to_tensor(infer_seq_mask),
            }
            final_attention_outputs = model(final_inputs, training=False, return_attention=need_attention)

        predictions['purpose'].append(purpose_pred)
        predictions['destination'].append(dest_pred)
        predictions['depart'].append(depart_pred)
        predictions['arrive'].append(arrive_pred)
        predictions['mode'].append(mode_pred)
        predictions['continue'].append(continue_pred)
        used_origin_history.append(used_origin_idx.copy())

        true_dest = batch_np['y_dest_zone']
        for k in TOPK_VALUES:
            hits = np.any(topk_indices[:, :, :k] == true_dest[:, :, None], axis=-1).astype(np.float32)
            dest_hits[k].append(hits)

        if len(captured_case_people) < case_person_limit:
            for local_idx, person_id in enumerate(batch_persons):
                if int(person_id) in captured_case_people:
                    continue
                captured_case_people.add(int(person_id))
                seq_len = int(batch_np['seq_len'][local_idx])
                for step_idx in range(seq_len):
                    destination_topk_rows.append(
                        {
                            'person_id': int(person_id),
                            'household_id': int(batch_np['household_id'][local_idx]),
                            'step_index': int(step_idx),
                            'inference_mode': 'autoregressive' if autoregressive else 'teacher_forcing',
                            'true_dest_zone_idx': int(batch_np['y_dest_zone'][local_idx, step_idx]),
                            'pred_dest_zone_idx': int(dest_pred[local_idx, step_idx]),
                            'top5_zone_idx': '|'.join(map(str, topk_indices[local_idx, step_idx, :5].tolist())),
                        }
                    )
                    case_rows.append(
                        {
                            'person_id': int(person_id),
                            'household_id': int(batch_np['household_id'][local_idx]),
                            'step_index': int(step_idx),
                            'inference_mode': 'autoregressive' if autoregressive else 'teacher_forcing',
                            'true_purpose': int(batch_np['y_purpose'][local_idx, step_idx]),
                            'pred_purpose': int(purpose_pred[local_idx, step_idx]),
                            'true_dest_zone': int(batch_np['y_dest_zone'][local_idx, step_idx]),
                            'pred_dest_zone': int(dest_pred[local_idx, step_idx]),
                            'true_depart_bin24': int(batch_np['y_depart_bin24'][local_idx, step_idx]),
                            'pred_depart_bin24': int(depart_pred[local_idx, step_idx]),
                            'true_arrive_bin24': int(batch_np['y_arrive_bin24'][local_idx, step_idx]),
                            'pred_arrive_bin24': int(arrive_pred[local_idx, step_idx]),
                            'true_mode': int(batch_np['y_mode'][local_idx, step_idx]),
                            'pred_mode': int(mode_pred[local_idx, step_idx]),
                            'true_continue': int(batch_np['y_continue'][local_idx, step_idx]),
                            'pred_continue': int(continue_pred[local_idx, step_idx]),
                        }
                    )
                if len(captured_case_people) >= case_person_limit:
                    break

        if need_attention and 'attention_scores' in final_attention_outputs:
            scores = final_attention_outputs['attention_scores'].numpy()
            max_cases = min(len(batch_persons), max(0, 5 - len(attention_rows)))
            for local_idx in range(max_cases):
                person_id = int(batch_persons[local_idx])
                seq_len = int(batch_np['seq_len'][local_idx])
                mean_scores = scores[local_idx].mean(axis=0)
                for from_step in range(seq_len):
                    for to_step in range(seq_len):
                        attention_rows.append(
                            {
                                'person_id': person_id,
                                'inference_mode': 'autoregressive' if autoregressive else 'teacher_forcing',
                                'from_step': int(from_step),
                                'to_step': int(to_step),
                                'attention_score': float(mean_scores[from_step, to_step]),
                            }
                        )

    pred_arrays = {key: np.concatenate(value, axis=0) for key, value in predictions.items()}
    topk_arrays = {k: np.concatenate(value, axis=0) for k, value in dest_hits.items()}
    used_origin_idx = np.concatenate(used_origin_history, axis=0)

    origin_idx = used_origin_idx
    dest_idx = pred_arrays['destination']
    origin_lon = zone_coords[origin_idx, 0]
    origin_lat = zone_coords[origin_idx, 1]
    dest_lon = zone_coords[dest_idx, 0]
    dest_lat = zone_coords[dest_idx, 1]
    pred_distance_km = haversine_km_np(origin_lon, origin_lat, dest_lon, dest_lat)

    purpose_support = zone_support_binary[dest_idx, np.clip(pred_arrays['purpose'], 0, zone_support_binary.shape[1] - 1)]
    home_zone_idx = arrays['static_cat'][:, 5][:, None]
    home_dest_match = (pred_arrays['destination'] == home_zone_idx).astype(np.float32)

    return {
        'predictions': pred_arrays,
        'destination_topk_hits': topk_arrays,
        'pred_distance_km': pred_distance_km,
        'purpose_support': purpose_support,
        'home_dest_match': home_dest_match,
        'case_rows': case_rows,
        'destination_topk_rows': destination_topk_rows,
        'attention_rows': attention_rows,
        'used_origin_idx': used_origin_idx,
        'inference_mode': 'autoregressive' if autoregressive else 'teacher_forcing',
    }

def summarize_chain_metrics(arrays: dict[str, np.ndarray], pred: dict[str, np.ndarray]) -> pd.DataFrame:
    seq_mask = arrays['seq_mask'] > 0

    def exact_rate(mask: np.ndarray, true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
        valid_any = mask.sum(axis=1) > 0
        if not np.any(valid_any):
            return float('nan')
        exact = np.all((~mask) | (true_vals == pred_vals), axis=1)
        return float(exact[valid_any].mean())

    task_masks = {
        'purpose': arrays['mask_purpose'] > 0,
        'destination': arrays['mask_dest_zone'] > 0,
        'depart': arrays['mask_depart_bin24'] > 0,
        'arrive': arrays['mask_arrive_bin24'] > 0,
        'mode': arrays['mask_mode'] > 0,
        'continue': arrays['mask_continue'] > 0,
    }
    task_truth = {
        'purpose': arrays['y_purpose'],
        'destination': arrays['y_dest_zone'],
        'depart': arrays['y_depart_bin24'],
        'arrive': arrays['y_arrive_bin24'],
        'mode': arrays['y_mode'],
        'continue': arrays['y_continue'],
    }
    task_pred = {
        'purpose': pred['purpose'],
        'destination': pred['destination'],
        'depart': pred['depart'],
        'arrive': pred['arrive'],
        'mode': pred['mode'],
        'continue': pred['continue'],
    }

    all_task_mask = task_masks['purpose'] | task_masks['destination'] | task_masks['depart'] | task_masks['arrive'] | task_masks['mode'] | task_masks['continue']
    all_task_exact = (
        ((~task_masks['purpose']) | (task_truth['purpose'] == task_pred['purpose']))
        & ((~task_masks['destination']) | (task_truth['destination'] == task_pred['destination']))
        & ((~task_masks['depart']) | (task_truth['depart'] == task_pred['depart']))
        & ((~task_masks['arrive']) | (task_truth['arrive'] == task_pred['arrive']))
        & ((~task_masks['mode']) | (task_truth['mode'] == task_pred['mode']))
        & ((~task_masks['continue']) | (task_truth['continue'] == task_pred['continue']))
    )
    valid_any = all_task_mask.sum(axis=1) > 0
    chain_all_exact = float(np.all(all_task_exact, axis=1)[valid_any].mean()) if np.any(valid_any) else float('nan')

    rows = [{
        'chain_exact_purpose_rate': exact_rate(task_masks['purpose'], task_truth['purpose'], task_pred['purpose']),
        'chain_exact_destination_rate': exact_rate(task_masks['destination'], task_truth['destination'], task_pred['destination']),
        'chain_exact_depart_rate': exact_rate(task_masks['depart'], task_truth['depart'], task_pred['depart']),
        'chain_exact_arrive_rate': exact_rate(task_masks['arrive'], task_truth['arrive'], task_pred['arrive']),
        'chain_exact_mode_rate': exact_rate(task_masks['mode'], task_truth['mode'], task_pred['mode']),
        'chain_exact_continue_rate': exact_rate(task_masks['continue'], task_truth['continue'], task_pred['continue']),
        'chain_exact_all_tasks_rate': chain_all_exact,
        'mean_seq_len': float(arrays['seq_len'].mean()),
    }]

    for prefix in (2, 3, 4):
        prefix_mask = np.zeros_like(seq_mask, dtype=bool)
        prefix_mask[:, : min(prefix, seq_mask.shape[1])] = seq_mask[:, : min(prefix, seq_mask.shape[1])]
        prefix_exact = (
            ((~prefix_mask) | (~task_masks['purpose']) | (task_truth['purpose'] == task_pred['purpose']))
            & ((~prefix_mask) | (~task_masks['destination']) | (task_truth['destination'] == task_pred['destination']))
            & ((~prefix_mask) | (~task_masks['depart']) | (task_truth['depart'] == task_pred['depart']))
            & ((~prefix_mask) | (~task_masks['arrive']) | (task_truth['arrive'] == task_pred['arrive']))
            & ((~prefix_mask) | (~task_masks['mode']) | (task_truth['mode'] == task_pred['mode']))
            & ((~prefix_mask) | (~task_masks['continue']) | (task_truth['continue'] == task_pred['continue']))
        )
        valid_prefix = prefix_mask.sum(axis=1) > 0
        rows[0][f'prefix{prefix}_all_tasks_exact_rate'] = float(np.all(prefix_exact, axis=1)[valid_prefix].mean()) if np.any(valid_prefix) else float('nan')

    return pd.DataFrame(rows)


def summarize_metrics(
    split_name: str,
    arrays: dict[str, np.ndarray],
    inference: dict[str, Any],
    home_purpose_id: int,
    time_bin_size_hours: int,
    mode_eval_labels: list[int],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = inference['predictions']
    topk_hits = inference['destination_topk_hits']

    metrics = {
        'split': split_name,
        'inference_mode': inference['inference_mode'],
        'purpose_acc': masked_accuracy(arrays['y_purpose'], pred['purpose'], arrays['mask_purpose']),
        'purpose_macro_f1': decode_metrics_f1(arrays['y_purpose'][arrays['mask_purpose'] > 0], pred['purpose'][arrays['mask_purpose'] > 0], [1, 2, 3, 4, 5, 6]),
        'destination_top1': topk_accuracy(topk_hits[1], arrays['mask_dest_zone']),
        'destination_top5': topk_accuracy(topk_hits[5], arrays['mask_dest_zone']),
        'destination_top10': topk_accuracy(topk_hits[10], arrays['mask_dest_zone']),
        'destination_top20': topk_accuracy(topk_hits[20], arrays['mask_dest_zone']),
        'depart_acc': masked_accuracy(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
        'arrive_acc': masked_accuracy(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
        'depart_mae_bins': masked_mae(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
        'arrive_mae_bins': masked_mae(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
        'depart_mae_hours': masked_mae(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']) * time_bin_size_hours,
        'arrive_mae_hours': masked_mae(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']) * time_bin_size_hours,
        'mode_acc': masked_accuracy(arrays['y_mode'], pred['mode'], arrays['mask_mode']),
        'mode_macro_f1': decode_metrics_f1(arrays['y_mode'][arrays['mask_mode'] > 0], pred['mode'][arrays['mask_mode'] > 0], mode_eval_labels),
        'continue_acc': masked_accuracy(arrays['y_continue'], pred['continue'], arrays['mask_continue']),
    }

    by_step_rows = []
    for step_idx in range(arrays['seq_mask'].shape[1]):
        by_step_rows.append(
            {
                'split': split_name,
                'step_index': step_idx,
                'n_samples': int(arrays['seq_mask'][:, step_idx].sum()),
                'purpose_acc': masked_accuracy(arrays['y_purpose'][:, step_idx], pred['purpose'][:, step_idx], arrays['mask_purpose'][:, step_idx]),
                'destination_top1': topk_accuracy(topk_hits[1][:, step_idx], arrays['mask_dest_zone'][:, step_idx]),
                'destination_top10': topk_accuracy(topk_hits[10][:, step_idx], arrays['mask_dest_zone'][:, step_idx]),
                'depart_acc': masked_accuracy(arrays['y_depart_bin24'][:, step_idx], pred['depart'][:, step_idx], arrays['mask_depart_bin24'][:, step_idx]),
                'arrive_acc': masked_accuracy(arrays['y_arrive_bin24'][:, step_idx], pred['arrive'][:, step_idx], arrays['mask_arrive_bin24'][:, step_idx]),
                'depart_mae_hours': masked_mae(arrays['y_depart_bin24'][:, step_idx], pred['depart'][:, step_idx], arrays['mask_depart_bin24'][:, step_idx]) * time_bin_size_hours,
                'arrive_mae_hours': masked_mae(arrays['y_arrive_bin24'][:, step_idx], pred['arrive'][:, step_idx], arrays['mask_arrive_bin24'][:, step_idx]) * time_bin_size_hours,
                'mode_acc': masked_accuracy(arrays['y_mode'][:, step_idx], pred['mode'][:, step_idx], arrays['mask_mode'][:, step_idx]),
                'continue_acc': masked_accuracy(arrays['y_continue'][:, step_idx], pred['continue'][:, step_idx], arrays['mask_continue'][:, step_idx]),
            }
        )
    by_step_df = pd.DataFrame(by_step_rows)

    topk_rows = []
    for k in TOPK_VALUES:
        topk_rows.append({'split': split_name, 'k': k, 'step_index': -1, 'topk_accuracy': topk_accuracy(topk_hits[k], arrays['mask_dest_zone'])})
        for step_idx in range(arrays['seq_mask'].shape[1]):
            topk_rows.append(
                {
                    'split': split_name,
                    'k': k,
                    'step_index': step_idx,
                    'topk_accuracy': topk_accuracy(topk_hits[k][:, step_idx], arrays['mask_dest_zone'][:, step_idx]),
                }
            )
    topk_df = pd.DataFrame(topk_rows)

    same_step_valid = (arrays['mask_depart_bin24'] > 0) & (arrays['mask_arrive_bin24'] > 0)
    adjacent_valid = (arrays['seq_mask'][:, 1:] > 0) & (arrays['seq_mask'][:, :-1] > 0)
    pred_depart = pred['depart']
    pred_arrive = pred['arrive']
    same_step_violation = ((pred_arrive < pred_depart) & same_step_valid).astype(np.float32)
    adjacent_violation = ((pred_depart[:, 1:] < pred_arrive[:, :-1]) & adjacent_valid).astype(np.float32)

    pred_distance = inference['pred_distance_km']
    pred_mode = pred['mode']
    walk_violation = ((pred_mode == 1) & (pred_distance > 3.0) & (arrays['mask_mode'] > 0)).astype(np.float32)
    bike_violation = ((pred_mode == 2) & (pred_distance > 8.0) & (arrays['mask_mode'] > 0)).astype(np.float32)
    home_mask = ((pred['purpose'] == home_purpose_id) & (arrays['mask_purpose'] > 0)).astype(np.float32)
    support_mask = ((pred['purpose'] != home_purpose_id) & (arrays['mask_purpose'] > 0) & (arrays['mask_dest_zone'] > 0)).astype(np.float32)

    behavior_df = pd.DataFrame(
        [
            {
                'split': split_name,
                'same_step_time_violation_rate': float(same_step_violation[same_step_valid].mean()) if same_step_valid.sum() else float('nan'),
                'adjacent_time_violation_rate': float(adjacent_violation[adjacent_valid].mean()) if adjacent_valid.sum() else float('nan'),
                'home_purpose_home_dest_rate': float((inference['home_dest_match'] * home_mask)[home_mask > 0].mean()) if home_mask.sum() else float('nan'),
                'purpose_destination_support_rate': float((inference['purpose_support'] * support_mask)[support_mask > 0].mean()) if support_mask.sum() else float('nan'),
                'long_walk_violation_rate': float(walk_violation[(pred_mode == 1) & (arrays['mask_mode'] > 0)].mean()) if np.any((pred_mode == 1) & (arrays['mask_mode'] > 0)) else float('nan'),
                'long_bike_violation_rate': float(bike_violation[(pred_mode == 2) & (arrays['mask_mode'] > 0)].mean()) if np.any((pred_mode == 2) & (arrays['mask_mode'] > 0)) else float('nan'),
            }
        ]
    )

    distance_rows = []
    valid_mode = arrays['mask_mode'] > 0
    flat_distance = pred_distance[valid_mode]
    flat_mode = pred_mode[valid_mode]
    bins = np.digitize(flat_distance, DISTANCE_BINS[1:-1], right=False)
    for bin_idx, label in enumerate(DISTANCE_BIN_LABELS):
        bin_mask = bins == bin_idx
        if not np.any(bin_mask):
            continue
        for mode_id in np.unique(flat_mode[bin_mask]):
            distance_rows.append(
                {
                    'split': split_name,
                    'distance_bin': label,
                    'mode_id': int(mode_id),
                    'count': int(np.sum(flat_mode[bin_mask] == mode_id)),
                    'share': float(np.mean(flat_mode[bin_mask] == mode_id)),
                }
            )
    distance_df = pd.DataFrame(distance_rows)

    purpose_rows = []
    support = inference['purpose_support']
    for purpose_id in np.unique(pred['purpose'][arrays['mask_purpose'] > 0]):
        if purpose_id == home_purpose_id:
            continue
        purpose_mask = (pred['purpose'] == purpose_id) & (arrays['mask_purpose'] > 0) & (arrays['mask_dest_zone'] > 0)
        if purpose_mask.sum() == 0:
            continue
        purpose_rows.append(
            {
                'split': split_name,
                'purpose_id': int(purpose_id),
                'samples': int(purpose_mask.sum()),
                'support_rate': float(support[purpose_mask].mean()),
            }
        )
    purpose_df = pd.DataFrame(purpose_rows)
    chain_df = summarize_chain_metrics(arrays, pred)
    chain_df.insert(0, 'split', split_name)
    return metrics, by_step_df, topk_df, behavior_df, distance_df, purpose_df, chain_df


def run_epoch(
    model: BehaviorStructuredACTNN,
    optimizer: tf.keras.optimizers.Optimizer,
    dataset: tf.data.Dataset,
    class_weights: dict[str, tf.Tensor],
    loss_weights: dict[str, float],
    temporal_lambda: float,
    time_reg_lambda: float,
    ss_prob: float,
) -> dict[str, float]:
    loss_sums: dict[str, float] = {}
    steps = 0
    for batch in dataset:
        batch_for_train = prepare_schedule_sampling_batch(model, batch, ss_prob)
        loss_parts = train_step(model, optimizer, batch_for_train, class_weights, loss_weights, temporal_lambda, time_reg_lambda)
        collect_loss_history_item(loss_sums, loss_parts)
        steps += 1
    return {key: value / max(steps, 1) for key, value in loss_sums.items()}


def evaluate_loss(
    model: BehaviorStructuredACTNN,
    dataset: tf.data.Dataset,
    class_weights: dict[str, tf.Tensor],
    loss_weights: dict[str, float],
    temporal_lambda: float,
    time_reg_lambda: float,
) -> dict[str, float]:
    loss_sums: dict[str, float] = {}
    steps = 0
    for batch in dataset:
        _, loss_parts, _ = compute_total_loss(model, batch, class_weights, loss_weights, temporal_lambda, time_reg_lambda, training=False)
        collect_loss_history_item(loss_sums, loss_parts)
        steps += 1
    return {key: value / max(steps, 1) for key, value in loss_sums.items()}


def append_experiment_registry(run_dir: Path, config: dict[str, Any], best_epoch: int, test_metrics: dict[str, float]) -> None:
    row = {
        'run_name': run_dir.name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_epoch': best_epoch,
        **{f'cfg_{k}': v for k, v in config.items()},
        **{f'test_{k}': v for k, v in test_metrics.items() if k != 'split'},
    }
    header = list(row.keys())
    write_header = not EXPERIMENT_REGISTRY.exists()
    with EXPERIMENT_REGISTRY.open('a', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_model_and_support(config: dict[str, Any], vocab: dict[str, Any], time_priors: dict[str, np.ndarray], time_behavior_priors: dict[str, np.ndarray], purpose_step_priors: np.ndarray, home_return_step_priors: np.ndarray, continue_home_priors: np.ndarray, mode_priors: dict[str, np.ndarray], zone_support_binary: np.ndarray, zone_coords: np.ndarray, zone_idx_to_id: dict[int, int], zone_purpose_matrix: np.ndarray) -> tuple[BehaviorStructuredACTNN, np.ndarray, np.ndarray, dict[int, int]]:
    zone_npz = np.load(MODEL_DATA_DIR / 'zone_feature_matrix.npz')
    zone_features = zone_npz['features'].astype(np.float32)
    num_zones = zone_features.shape[0] + 1
    padded_zone_features = np.zeros((num_zones, zone_features.shape[1]), dtype=np.float32)
    padded_zone_features[1:] = zone_features

    if zone_purpose_matrix.shape[0] < num_zones:
        pad_rows = num_zones - zone_purpose_matrix.shape[0]
        zone_purpose_matrix = np.vstack([zone_purpose_matrix, np.zeros((pad_rows, zone_purpose_matrix.shape[1]), dtype=np.float32)])
        zone_support_binary = np.vstack([zone_support_binary, np.zeros((pad_rows, zone_support_binary.shape[1]), dtype=np.float32)])
        zone_coords = np.vstack([zone_coords, np.zeros((pad_rows, zone_coords.shape[1]), dtype=np.float32)])

    num_time = int(config['num_time_classes']) + 1
    model = BehaviorStructuredACTNN(
        num_zones=num_zones,
        num_purpose=len(vocab['purpose_target_vocab']),
        num_time=num_time,
        num_gap=NUM_GAP_CLASSES,
        num_mode=len(vocab['mode_target_vocab']),
        num_gender=len(vocab['gender_vocab']),
        num_occupation=len(vocab['occupation_vocab']),
        num_schooling=len(vocab['schooling_status_vocab']),
        num_housing=len(vocab['housing_tenure_vocab']),
        num_usual_commute=len(vocab['usual_commute_mode_vocab']),
        num_district=len(vocab['origin_admin_district_vocab']),
        zone_feature_matrix=padded_zone_features,
        zone_purpose_matrix=zone_purpose_matrix,
        zone_coord_matrix=zone_coords.astype(np.float32),
        purpose_step_prior_matrix=purpose_step_priors,
        home_return_step_prior=home_return_step_priors,
        purpose_depart_prior_matrix=time_behavior_priors['purpose_depart'],
        purpose_duration_prior_matrix=time_behavior_priors['purpose_duration'],
        mode_duration_prior_matrix=time_behavior_priors['mode_duration'],
        first_depart_prior_matrix=time_priors['first_depart'],
        gap_prior_matrix=time_priors['gap'],
        duration_prior_matrix=time_priors['duration'],
        continue_prior_matrix=time_priors['continue'],
        continue_home_prior_matrix=continue_home_priors,
        mode_step_prior_matrix=mode_priors['step'],
        purpose_mode_prior_matrix=mode_priors['purpose'],
        mode_distance_prior_matrix=mode_priors['distance'],
        mode_transition_prior_matrix=mode_priors['transition'],
        mode_usual_commute_prior_matrix=mode_priors['usual_commute'],
        home_purpose_id=int(vocab['purpose_target_vocab']['Home']),
        bos_purpose_id=int(vocab['bos_purpose_id']),
        bos_mode_id=int(vocab['bos_mode_id']),
        config=config,
    )
    return model, zone_support_binary, zone_coords, zone_idx_to_id


def save_case_outputs(run_dir: Path, inference: dict[str, Any], zone_idx_to_id: dict[int, int]) -> None:
    case_df = pd.DataFrame(inference['case_rows'])
    if not case_df.empty:
        case_df['true_dest_zone_id'] = case_df['true_dest_zone'].map(zone_idx_to_id)
        case_df['pred_dest_zone_id'] = case_df['pred_dest_zone'].map(zone_idx_to_id)
        case_df.to_csv(run_dir / 'figure_data' / 'case_predictions.csv', index=False, encoding='utf-8-sig')
    topk_df = pd.DataFrame(inference['destination_topk_rows'])
    if not topk_df.empty:
        topk_df['true_dest_zone_id'] = topk_df['true_dest_zone_idx'].map(zone_idx_to_id)
        topk_df['pred_dest_zone_id'] = topk_df['pred_dest_zone_idx'].map(zone_idx_to_id)
        topk_df.to_csv(run_dir / 'figure_data' / 'destination_predictions_sample.csv', index=False, encoding='utf-8-sig')
    attn_df = pd.DataFrame(inference['attention_rows'])
    if not attn_df.empty:
        attn_df.to_csv(run_dir / 'figure_data' / 'attention_weights_case.csv', index=False, encoding='utf-8-sig')

def main() -> None:
    parser = argparse.ArgumentParser(description='Train the new ACT-NN model on model_data NPZ datasets.')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--zone-embed-dim', type=int, default=32)
    parser.add_argument('--zone-fixed-dim', type=int, default=24)
    parser.add_argument('--general-embed-dim', type=int, default=8)
    parser.add_argument('--use-step-embedding', action='store_true')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--temporal-lambda', type=float, default=0.15)
    parser.add_argument('--time-reg-lambda', type=float, default=0.0)
    parser.add_argument('--destination-loss-weight', type=float, default=1.35)
    parser.add_argument('--time-bin-size-hours', type=int, default=DEFAULT_TIME_BIN_SIZE_HOURS)
    parser.add_argument('--mode-schema', type=str, default='fine7', choices=['fine7', 'coarse4', 'standard3'])
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--schedule-sampling-max', type=float, default=0.35)
    parser.add_argument('--schedule-sampling-warmup', type=int, default=4)
    parser.add_argument('--subset-train', type=int, default=None)
    parser.add_argument('--subset-valid', type=int, default=None)
    parser.add_argument('--subset-test', type=int, default=None)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime('actnn_%Y%m%d_%H%M%S')
    run_dir = make_run_dir(run_name)
    set_global_seed(args.seed)

    vocab = load_json(MODEL_DATA_DIR / 'category_vocabularies.json')
    dataset_summary = load_json(MODEL_DATA_DIR / 'dataset_summary.json')

    num_time_classes = compute_num_time_classes(args.time_bin_size_hours)

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'zone_embed_dim': args.zone_embed_dim,
        'zone_fixed_dim': args.zone_fixed_dim,
        'general_embed_dim': args.general_embed_dim,
        'use_step_embedding': args.use_step_embedding,
        'num_heads': args.num_heads,
        'time_prior_init': 0.8,
        'continue_prior_init': 1.2,
        'continue_behavior_init': 1.0,
        'mode_prior_init': 0.35 if args.mode_schema == 'fine7' else 1.1,
        'mode_distance_prior_init': 0.6 if args.mode_schema == 'fine7' else 1.8,
        'mode_transition_prior_init': 0.9 if args.mode_schema == 'fine7' else 0.5,
        'mode_usual_commute_prior_init': 0.9 if args.mode_schema == 'fine7' else 0.5,
        'temporal_lambda': args.temporal_lambda,
        'time_reg_lambda': args.time_reg_lambda,
        'destination_loss_weight': args.destination_loss_weight,
        'time_bin_size_hours': args.time_bin_size_hours,
        'mode_schema': args.mode_schema,
        'num_time_classes': num_time_classes,
        'gap_min': GAP_MIN,
        'gap_max': GAP_MAX,
        'patience': args.patience,
        'seed': args.seed,
        'schedule_sampling_max': args.schedule_sampling_max,
        'schedule_sampling_warmup': args.schedule_sampling_warmup,
        'subset_train': args.subset_train,
        'subset_valid': args.subset_valid,
        'subset_test': args.subset_test,
        'max_steps': dataset_summary['max_steps'],
        'valid_test_inference_mode': 'autoregressive',
        'walk_mode_id': int(vocab['mode_target_vocab'].get('Walk', 0)),
        'bike_mode_id': int(vocab['mode_target_vocab'].get('BikeEbike', 0)),
        'bus_mode_id': int(vocab['mode_target_vocab'].get('Bus', 0)),
        'metro_mode_id': int(vocab['mode_target_vocab'].get('Metro', 0)),
        'taxi_mode_id': int(vocab['mode_target_vocab'].get('TaxiRidehail', 0)),
        'car_mode_id': int(vocab['mode_target_vocab'].get('CarMotor', 0)),
    }
    (run_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    mode_vocab, mode_old_to_new, new_bos_mode_id = build_mode_schema(vocab['mode_target_vocab'], args.mode_schema, int(vocab['bos_mode_id']))
    vocab = dict(vocab)
    vocab['mode_target_vocab'] = mode_vocab
    vocab['bos_mode_id'] = int(new_bos_mode_id)
    mode_eval_labels = sorted(v for k, v in mode_vocab.items() if v > 0)

    train_arrays = remap_mode_arrays(remap_time_arrays(load_split_arrays('train', args.subset_train), args.time_bin_size_hours), mode_old_to_new, new_bos_mode_id)
    valid_arrays = remap_mode_arrays(remap_time_arrays(load_split_arrays('valid', args.subset_valid), args.time_bin_size_hours), mode_old_to_new, new_bos_mode_id)
    test_arrays = remap_mode_arrays(remap_time_arrays(load_split_arrays('test', args.subset_test), args.time_bin_size_hours), mode_old_to_new, new_bos_mode_id)

    gap_labels_train, gap_mask_train = derive_gap_targets_from_arrays(
        train_arrays['y_depart_bin24'],
        train_arrays['y_arrive_bin24'],
        train_arrays['mask_depart_bin24'],
        train_arrays['mask_arrive_bin24'],
    )
    duration_labels_train, duration_mask_train = derive_duration_targets_from_arrays(
        train_arrays['y_depart_bin24'],
        train_arrays['y_arrive_bin24'],
        train_arrays['mask_depart_bin24'],
        train_arrays['mask_arrive_bin24'],
        num_time_classes,
    )
    mode_family_lookup_np = build_mode_family_lookup(vocab['mode_target_vocab'])
    mode_family_labels_train = mode_family_lookup_np[train_arrays['y_mode'].astype(np.int32)]
    class_weights = {
        'purpose': tf.constant(compute_class_weights(train_arrays['y_purpose'], train_arrays['mask_purpose'], len(vocab['purpose_target_vocab'])), dtype=tf.float32),
        'depart_first': tf.constant(np.ones(num_time_classes + 1, dtype=np.float32), dtype=tf.float32),
        'depart_gap': tf.constant(np.ones(NUM_GAP_CLASSES, dtype=np.float32), dtype=tf.float32),
        'arrive': tf.constant(np.ones(num_time_classes + 1, dtype=np.float32), dtype=tf.float32),
        'mode': tf.constant(compute_class_weights(train_arrays['y_mode'], train_arrays['mask_mode'], len(vocab['mode_target_vocab']), power=0.65 if args.mode_schema == 'fine7' else 0.35), dtype=tf.float32),
        'mode_family': tf.constant(compute_class_weights(mode_family_labels_train, train_arrays['mask_mode'], 4, power=0.25), dtype=tf.float32),
        'continue': tf.constant(compute_class_weights(train_arrays['y_continue'], train_arrays['mask_continue'], 2), dtype=tf.float32),
    }
    loss_weights = {'purpose': 1.0, 'destination': args.destination_loss_weight, 'depart': 1.2, 'arrive': 1.2, 'mode': 2.4 if args.mode_schema == 'fine7' else 1.8, 'mode_family': 0.0, 'continue': 0.6}

    zone_purpose_matrix, zone_support_binary, zone_coords, zone_idx_to_id = prepare_zone_support()
    time_priors = prepare_step_time_priors(train_arrays, num_time_classes)
    time_behavior_priors = prepare_time_behavior_priors(train_arrays, len(vocab['purpose_target_vocab']), len(vocab['mode_target_vocab']), num_time_classes)
    purpose_step_priors = prepare_purpose_step_priors(train_arrays, len(vocab['purpose_target_vocab']))
    home_return_step_priors = prepare_home_return_step_priors(train_arrays)
    continue_home_priors = prepare_continue_home_priors(train_arrays)
    mode_priors = prepare_mode_priors(train_arrays, len(vocab['purpose_target_vocab']), len(vocab['mode_target_vocab']))
    mode_priors['distance'] = prepare_mode_distance_priors(train_arrays, zone_coords, len(vocab['mode_target_vocab']))
    mode_priors['transition'] = prepare_mode_transition_priors(train_arrays, len(vocab['mode_target_vocab']) + 1, len(vocab['mode_target_vocab']))
    mode_priors['usual_commute'] = prepare_mode_usual_commute_priors(train_arrays, len(vocab['usual_commute_mode_vocab']), len(vocab['mode_target_vocab']))
    model, zone_support_binary, zone_coords, zone_idx_to_id = build_model_and_support(
        config, vocab, time_priors, time_behavior_priors, purpose_step_priors, home_return_step_priors, continue_home_priors, mode_priors, zone_support_binary, zone_coords, zone_idx_to_id, zone_purpose_matrix
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    dummy_batch = {key: tf.convert_to_tensor(value[:2]) for key, value in train_arrays.items()}
    _ = model(split_inputs(dummy_batch), training=False)

    train_ds = build_tf_dataset(train_arrays, args.batch_size, shuffle=True)
    valid_ds = build_tf_dataset(valid_arrays, args.batch_size, shuffle=False)

    checkpoint = tf.train.Checkpoint(model=model)
    best_ckpt_prefix = str(run_dir / 'checkpoints' / 'best_ckpt')

    history_rows = []
    best_val = math.inf
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        ss_prob = compute_schedule_sampling_prob(epoch, args.epochs, args.schedule_sampling_warmup, args.schedule_sampling_max)
        train_loss = run_epoch(model, optimizer, train_ds, class_weights, loss_weights, args.temporal_lambda, args.time_reg_lambda, ss_prob)
        valid_loss = evaluate_loss(model, valid_ds, class_weights, loss_weights, args.temporal_lambda, args.time_reg_lambda)
        row = {'epoch': epoch, 'schedule_sampling_prob': ss_prob}
        row.update({f'train_{k}': v for k, v in train_loss.items()})
        row.update({f'valid_{k}': v for k, v in valid_loss.items()})
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / 'epoch_metrics.csv', index=False, encoding='utf-8-sig')

        print(
            f"Epoch {epoch:02d} | ss_prob={ss_prob:.3f} | train_total={train_loss['total']:.4f} | valid_total={valid_loss['total']:.4f} | "
            f"train_dest={train_loss['destination']:.4f} | valid_dest={valid_loss['destination']:.4f}"
        )

        if valid_loss['total'] < best_val:
            best_val = valid_loss['total']
            best_epoch = epoch
            patience_left = args.patience
            checkpoint.write(best_ckpt_prefix)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'Early stopping at epoch {epoch}.')
                break

    checkpoint.restore(best_ckpt_prefix).expect_partial()

    results = []
    figure_frames: dict[str, pd.DataFrame] = {}
    for split_name, arrays in [('train', train_arrays), ('valid', valid_arrays), ('test', test_arrays)]:
        use_autoregressive = split_name in {'valid', 'test'}
        inference = forward_batches(
            model=model,
            arrays=arrays,
            batch_size=args.batch_size,
            home_purpose_id=int(vocab['purpose_target_vocab']['Home']),
            zone_support_binary=zone_support_binary,
            zone_coords=zone_coords,
            case_person_limit=40 if split_name == 'test' else 10,
            autoregressive=use_autoregressive,
        )
        metrics, by_step_df, topk_df, behavior_df, distance_df, purpose_df, chain_df = summarize_metrics(
            split_name, arrays, inference, int(vocab['purpose_target_vocab']['Home']), args.time_bin_size_hours, mode_eval_labels
        )
        results.append(metrics)
        figure_frames[f'{split_name}_metrics_by_step'] = by_step_df
        figure_frames[f'{split_name}_destination_topk'] = topk_df
        figure_frames[f'{split_name}_behavioral_consistency'] = behavior_df
        if not distance_df.empty:
            figure_frames[f'{split_name}_mode_by_distance_bin'] = distance_df
        if not purpose_df.empty:
            figure_frames[f'{split_name}_purpose_destination_compatibility'] = purpose_df
        if not chain_df.empty:
            figure_frames[f'{split_name}_chain_level_metrics'] = chain_df
        if split_name == 'test':
            save_case_outputs(run_dir, inference, zone_idx_to_id)

    pd.DataFrame(results).to_csv(run_dir / 'test_metrics_overall.csv', index=False, encoding='utf-8-sig')
    for name, frame in figure_frames.items():
        frame.to_csv(run_dir / 'figure_data' / f'{name}.csv', index=False, encoding='utf-8-sig')

    append_experiment_registry(run_dir, config, best_epoch, next(row for row in results if row['split'] == 'test'))
    print(pd.DataFrame(results).to_string(index=False))
    print(f'Best epoch: {best_epoch}')
    print(f'Outputs saved to: {run_dir}')


if __name__ == '__main__':
    main()


















