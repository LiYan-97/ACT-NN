from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from actnn_tour_graph_model import TourInterpretableGraphACTNN
from train_actnn import (
    EXPERIMENT_REGISTRY,
    MODEL_DATA_DIR,
    OUTPUT_ROOT,
    TOPK_VALUES,
    append_experiment_registry,
    build_mode_schema,
    remap_mode_arrays,
    remap_time_arrays,
    load_json,
    make_run_dir,
    load_split_arrays,
    compute_num_time_classes,
    compute_class_weights,
    build_tf_dataset,
    split_inputs,
    masked_sparse_ce,
    masked_accuracy,
    masked_huber,
    masked_mae,
    topk_accuracy,
    decode_metrics_f1,
    compute_step_log_prior_matrix,
    compute_conditional_log_prior,
    prepare_zone_support,
    prepare_mode_priors,
    prepare_mode_distance_priors,
    prepare_mode_transition_priors,
    prepare_mode_usual_commute_priors,
    set_global_seed,
    save_case_outputs,
    compute_schedule_sampling_prob,
)



def prepare_origin_candidate_log_mask(zone_coords: np.ndarray, k: int = 64, penalty: float = -2.5) -> np.ndarray:
    num_zones = zone_coords.shape[0]
    mask = np.full((num_zones, num_zones), float(penalty), dtype=np.float32)
    if num_zones == 0:
        return mask
    mask[0, :] = 0.0
    mask[:, 0] = 0.0
    coords = zone_coords.astype(np.float32)
    for i in range(1, num_zones):
        origin = coords[i]
        d2 = np.sum((coords - origin) ** 2, axis=1)
        kk = min(max(int(k), 1), num_zones)
        idx = np.argpartition(d2, kk - 1)[:kk]
        mask[i, idx] = 0.0
        mask[i, i] = 0.0
    return mask
def build_model_and_support(config: dict[str, Any], vocab: dict[str, Any], priors: dict[str, np.ndarray], zone_support_binary: np.ndarray, zone_coords: np.ndarray, zone_idx_to_id: dict[int, int], zone_purpose_matrix: np.ndarray) -> tuple[TourInterpretableGraphACTNN, np.ndarray, np.ndarray, dict[int, int]]:
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
    origin_candidate_log_mask = prepare_origin_candidate_log_mask(zone_coords.astype(np.float32), int(config.get('candidate_knn_k', 64)), float(config.get('candidate_penalty', -2.5)))
    model = TourInterpretableGraphACTNN(
        num_zones=num_zones,
        num_purpose=len(vocab['purpose_target_vocab']),
        num_time=num_time,
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
        purpose_step_prior_matrix=priors['purpose_step'],
        origin_destination_prior_matrix=priors['origin_destination'],
        depart_step_prior_matrix=priors['depart_step'],
        arrive_step_prior_matrix=priors['arrive_step'],
        purpose_depart_prior_matrix=priors['purpose_depart'],
        purpose_arrive_prior_matrix=priors['purpose_arrive'],
        mode_step_prior_matrix=priors['mode_step'],
        purpose_mode_prior_matrix=priors['mode_purpose'],
        mode_distance_prior_matrix=priors['mode_distance'],
        mode_transition_prior_matrix=priors['mode_transition'],
        mode_usual_commute_prior_matrix=priors['mode_usual_commute'],
        origin_candidate_log_mask=origin_candidate_log_mask,
        config=config,
    )
    return model, zone_support_binary, zone_coords, zone_idx_to_id



def compute_origin_destination_log_prior(
    origin_zone: np.ndarray,
    origin_mask: np.ndarray,
    dest_zone: np.ndarray,
    dest_mask: np.ndarray,
    num_zones: int,
    smoothing: float = 0.5,
) -> np.ndarray:
    counts = np.full((num_zones, num_zones), smoothing, dtype=np.float64)
    counts[:, 0] = 0.0
    valid = (origin_mask > 0) & (dest_mask > 0) & (origin_zone > 0) & (dest_zone > 0)
    if not np.any(valid):
        return np.zeros((num_zones, num_zones), dtype=np.float32)
    origin_flat = origin_zone[valid].astype(np.int64)
    dest_flat = dest_zone[valid].astype(np.int64)
    np.add.at(counts, (origin_flat, dest_flat), 1.0)
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    if num_zones > 1:
        log_probs[:, 1:] -= log_probs[:, 1:].mean(axis=1, keepdims=True)
    log_probs[0, :] = 0.0
    return log_probs.astype(np.float32)


def compute_zone_district_lookup(train_arrays: dict[str, np.ndarray], num_zones: int, num_district: int) -> np.ndarray:
    counts = np.zeros((num_zones, num_district), dtype=np.int64)
    origin_zone = train_arrays['step_cat'][:, :, 0].astype(np.int32)
    district = train_arrays['step_cat'][:, :, 3].astype(np.int32)
    valid = (train_arrays['seq_mask'] > 0) & (origin_zone > 0) & (district >= 0) & (district < num_district)
    if np.any(valid):
        np.add.at(counts, (origin_zone[valid], district[valid]), 1)
    lookup = counts.argmax(axis=1).astype(np.int32)
    lookup[0] = 0
    return lookup


def augment_main_targets(arrays: dict[str, np.ndarray], purpose_vocab: dict[str, int]) -> dict[str, np.ndarray]:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
    n, _ = out['seq_mask'].shape
    y_main_step = np.zeros(n, dtype=np.int32)
    y_main_purpose = np.zeros(n, dtype=np.int32)
    y_main_dest = np.zeros(n, dtype=np.int32)
    y_main_arrive = np.zeros(n, dtype=np.int32)
    y_main_depart = np.zeros(n, dtype=np.int32)
    y_main_mode = np.zeros(n, dtype=np.int32)
    mask_main = np.zeros(n, dtype=np.float32)

    prio = {
        int(purpose_vocab.get('Home', 1)): 0.0,
        int(purpose_vocab.get('Work', 2)): 500.0,
        int(purpose_vocab.get('School', 3)): 420.0,
        int(purpose_vocab.get('Shopping', 4)): 260.0,
        int(purpose_vocab.get('Visit', 5)): 220.0,
        int(purpose_vocab.get('OtherGoal', 6)): 180.0,
    }
    home_id = int(purpose_vocab.get('Home', 1))

    for i in range(n):
        valid_idx = np.where(out['seq_mask'][i] > 0)[0]
        if len(valid_idx) == 0:
            continue
        scores = []
        for pos in valid_idx:
            purpose_id = int(out['y_purpose'][i, pos])
            base = prio.get(purpose_id, 100.0)
            dwell = 0.0
            if pos < valid_idx[-1]:
                next_pos = pos + 1
                if out['mask_depart_bin24'][i, next_pos] > 0 and out['mask_arrive_bin24'][i, pos] > 0:
                    dwell = max(float(out['y_depart_bin24'][i, next_pos] - out['y_arrive_bin24'][i, pos]), 0.0)
            if purpose_id == home_id:
                base -= 100.0
            scores.append(base + dwell)
        main_pos = int(valid_idx[int(np.argmax(np.asarray(scores, dtype=np.float32)))])
        y_main_step[i] = main_pos
        y_main_purpose[i] = int(out['y_purpose'][i, main_pos])
        y_main_dest[i] = int(out['y_dest_zone'][i, main_pos])
        y_main_arrive[i] = int(out['y_arrive_bin24'][i, main_pos])
        y_main_depart[i] = int(out['y_depart_bin24'][i, main_pos])
        y_main_mode[i] = int(out['y_mode'][i, main_pos])
        mask_main[i] = 1.0

    out['y_main_step'] = y_main_step
    out['y_main_purpose'] = y_main_purpose
    out['y_main_dest_zone'] = y_main_dest
    out['y_main_arrive_bin24'] = y_main_arrive
    out['y_main_depart_bin24'] = y_main_depart
    out['y_main_mode'] = y_main_mode
    out['mask_main'] = mask_main
    return out


def augment_relation_targets(arrays: dict[str, np.ndarray], purpose_vocab: dict[str, int]) -> dict[str, np.ndarray]:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
    n, max_steps = out['seq_mask'].shape
    y_relation = np.zeros((n, max_steps), dtype=np.int32)
    mask_relation = np.zeros((n, max_steps), dtype=np.float32)
    home_id = int(purpose_vocab.get('Home', 1))
    main_step = out['y_main_step']
    main_dest = out['y_main_dest_zone']
    origin_zone = out['step_cat'][:, :, 0]
    dest_zone = out['y_dest_zone']
    purpose = out['y_purpose']
    for i in range(n):
        valid_idx = np.where(out['seq_mask'][i] > 0)[0]
        if len(valid_idx) == 0:
            continue
        m = int(main_step[i])
        md = int(main_dest[i])
        for pos in valid_idx:
            mask_relation[i, pos] = 1.0
            if pos == m:
                y_relation[i, pos] = 2
            elif pos < m:
                y_relation[i, pos] = 1
            elif purpose[i, pos] == home_id:
                y_relation[i, pos] = 5
            elif md > 0 and (dest_zone[i, pos] == md or origin_zone[i, pos] == md):
                y_relation[i, pos] = 3
            else:
                y_relation[i, pos] = 4
    out['y_relation_to_main'] = y_relation
    out['mask_relation_to_main'] = mask_relation
    return out


def augment_secondary_insertion_targets(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
    relation = out['y_relation_to_main']
    mask_relation = out['mask_relation_to_main']
    y_secondary = np.zeros_like(relation, dtype=np.int32)
    y_secondary[relation == 1] = 1  # before_main
    y_secondary[relation == 3] = 2  # around_main
    y_secondary[relation == 4] = 3  # after_main
    out['y_secondary_insert'] = y_secondary
    out['mask_secondary_insert'] = mask_relation.astype(np.float32)
    return out

def expected_time_from_logits(logits: tf.Tensor) -> tf.Tensor:
    probs = tf.nn.softmax(logits, axis=-1)
    bins = tf.range(tf.shape(logits)[-1], dtype=tf.float32) - 1.0
    bins = tf.maximum(bins, 0.0)
    return tf.reduce_sum(probs * bins[tf.newaxis, tf.newaxis, :], axis=-1)


def build_time_soft_targets(labels: tf.Tensor, mask: tf.Tensor, num_classes: int, sigma: float) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    mask = tf.cast(mask, tf.float32)
    if sigma <= 1e-8:
        return tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    centers = tf.maximum(labels - 1, 0)
    bins = tf.cast(tf.range(num_classes), tf.float32)[tf.newaxis, tf.newaxis, :]
    centers_f = tf.cast(centers, tf.float32)[..., tf.newaxis]
    dist = bins - centers_f
    weights = tf.exp(-0.5 * tf.square(dist / tf.cast(sigma, tf.float32)))
    valid_mask = tf.concat([tf.zeros((1,), dtype=tf.float32), tf.ones((num_classes - 1,), dtype=tf.float32)], axis=0)
    weights = weights * valid_mask[tf.newaxis, tf.newaxis, :]
    denom = tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-6
    soft = weights / denom
    one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    return mask[..., tf.newaxis] * soft + (1.0 - mask[..., tf.newaxis]) * one_hot


def masked_soft_ce(logits: tf.Tensor, soft_targets: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    losses = -tf.reduce_sum(soft_targets * tf.nn.log_softmax(logits, axis=-1), axis=-1)
    losses *= tf.cast(mask, tf.float32)
    denom = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(losses) / denom


def masked_ordinal_loss_from_probs(probs: tf.Tensor, labels: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    num_classes = tf.shape(probs)[-1]
    if probs.shape.rank != 3:
        raise ValueError('Expected probs rank 3 for ordinal loss.')
    if probs.shape[-1] is not None and probs.shape[-1] <= 2:
        return tf.constant(0.0, dtype=tf.float32)
    valid_probs = probs[..., 1:]
    tail_cumsum = tf.reverse(tf.math.cumsum(tf.reverse(valid_probs, axis=[-1]), axis=-1), axis=[-1])
    greater_probs = tail_cumsum[..., 1:]
    thresholds = tf.range(1, num_classes - 1, dtype=tf.int32)
    labels_clipped = tf.maximum(tf.cast(labels, tf.int32) - 1, 0)
    targets = tf.cast(labels_clipped[..., tf.newaxis] > thresholds[tf.newaxis, tf.newaxis, :], tf.float32)
    losses = tf.keras.backend.binary_crossentropy(targets, tf.clip_by_value(greater_probs, 1e-6, 1.0 - 1e-6))
    losses = tf.reduce_mean(losses, axis=-1) * tf.cast(mask, tf.float32)
    denom = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(losses) / denom


def masked_sparse_ce_vector(logits: tf.Tensor, labels: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits)
    losses *= tf.cast(mask, tf.float32)
    denom = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(losses) / denom


def within_one_bin_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if valid.sum() == 0:
        return float('nan')
    true_bin = np.maximum(true_labels.astype(np.int32) - 1, 0)
    pred_bin = np.maximum(pred_labels.astype(np.int32) - 1, 0)
    return float((np.abs(true_bin[valid] - pred_bin[valid]) <= 1).mean())


def map_mode_to_family_tf(labels: tf.Tensor) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    out = tf.zeros_like(labels, dtype=tf.int32)
    out = tf.where((labels == 1) | (labels == 2), tf.zeros_like(out), out)
    out = tf.where((labels == 3) | (labels == 4), tf.fill(tf.shape(out), 1), out)
    out = tf.where((labels == 5) | (labels == 6), tf.fill(tf.shape(out), 2), out)
    out = tf.where(labels == 7, tf.fill(tf.shape(out), 3), out)
    return out


def map_time_to_coarse_tf(labels: tf.Tensor, num_time_classes: int) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    valid = labels > 0
    bins0 = tf.maximum(labels - 1, 0)
    coarse = tf.clip_by_value((bins0 * 4) // max(num_time_classes, 1), 0, 3)
    return tf.where(valid, coarse, tf.zeros_like(labels, dtype=tf.int32))


def derive_duration_targets_from_batch(batch: dict[str, tf.Tensor], num_time_classes: int) -> tuple[tf.Tensor, tf.Tensor]:
    depart_bin = tf.maximum(tf.cast(batch['y_depart_bin24'], tf.int32) - 1, 0)
    arrive_bin = tf.maximum(tf.cast(batch['y_arrive_bin24'], tf.int32) - 1, 0)
    duration_bin = tf.clip_by_value(arrive_bin - depart_bin, 0, max(num_time_classes - 1, 0))
    duration_mask = tf.cast((batch['mask_depart_bin24'] > 0) & (batch['mask_arrive_bin24'] > 0), tf.float32)
    duration_label = tf.where(duration_mask > 0, duration_bin + 1, 0)
    return tf.cast(duration_label, tf.int32), duration_mask


def derive_dwell_targets_from_batch(batch: dict[str, tf.Tensor], num_time_classes: int) -> tuple[tf.Tensor, tf.Tensor]:
    arrive_bin = tf.maximum(tf.cast(batch['y_arrive_bin24'], tf.int32) - 1, 0)
    next_depart_bin = tf.maximum(tf.cast(batch['y_depart_bin24'][:, 1:], tf.int32) - 1, 0)
    current_arrive_bin = arrive_bin[:, :-1]
    dwell_bin = tf.clip_by_value(next_depart_bin - current_arrive_bin, 0, max(num_time_classes - 1, 0))
    dwell_mask_core = tf.cast((batch['mask_arrive_bin24'][:, :-1] > 0) & (batch['mask_depart_bin24'][:, 1:] > 0) & (batch['seq_mask'][:, 1:] > 0), tf.float32)
    dwell_label_core = tf.where(dwell_mask_core > 0, dwell_bin + 1, 0)
    pad_label = tf.zeros((tf.shape(arrive_bin)[0], 1), dtype=tf.int32)
    pad_mask = tf.zeros((tf.shape(arrive_bin)[0], 1), dtype=tf.float32)
    dwell_label = tf.concat([tf.cast(dwell_label_core, tf.int32), pad_label], axis=1)
    dwell_mask = tf.concat([dwell_mask_core, pad_mask], axis=1)
    return dwell_label, dwell_mask

def temporal_consistency_penalty(outputs: dict[str, tf.Tensor], seq_mask: tf.Tensor) -> tf.Tensor:
    depart_expect = expected_time_from_logits(outputs['depart_logits'])
    arrive_expect = expected_time_from_logits(outputs['arrive_logits'])
    same_step = tf.nn.relu(depart_expect - arrive_expect)
    pair_mask = seq_mask[:, 1:] * seq_mask[:, :-1]
    chain_penalty = tf.nn.relu(arrive_expect[:, :-1] - depart_expect[:, 1:] - 0.5)
    same_loss = tf.reduce_sum(same_step * seq_mask)
    chain_loss = tf.reduce_sum(chain_penalty * pair_mask)
    denom = tf.reduce_sum(seq_mask) + tf.reduce_sum(pair_mask) + 1e-6
    return (same_loss + chain_loss) / denom


def compute_ramp_factor(epoch: int, warmup: int, ramp: int) -> float:
    if epoch <= warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(min(max((epoch - warmup) / float(ramp), 0.0), 1.0))


def compute_total_loss(model: TourInterpretableGraphACTNN, batch: dict[str, tf.Tensor], class_weights: dict[str, tf.Tensor], loss_weights: dict[str, float], temporal_lambda: float, time_soft_sigma: float, time_distance_lambda: float, time_ordinal_lambda: float, training: bool) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    outputs = model(split_inputs(batch), training=training)
    y_mode_family = map_mode_to_family_tf(batch['y_mode'])
    y_arrive_coarse = map_time_to_coarse_tf(batch['y_arrive_bin24'], model.num_time - 1)
    y_depart_coarse = map_time_to_coarse_tf(batch['y_depart_bin24'], model.num_time - 1)
    duration_labels, duration_mask = derive_duration_targets_from_batch(batch, model.num_time - 1)
    dwell_labels, dwell_mask = derive_dwell_targets_from_batch(batch, model.num_time - 1)
    arrive_expect = expected_time_from_logits(outputs['arrive_logits'])
    depart_expect = expected_time_from_logits(outputs['depart_logits'])
    duration_expect = expected_time_from_logits(outputs['duration_logits'])
    dwell_expect = expected_time_from_logits(outputs['dwell_logits'])
    arrive_true = tf.maximum(tf.cast(batch['y_arrive_bin24'], tf.float32) - 1.0, 0.0)
    depart_true = tf.maximum(tf.cast(batch['y_depart_bin24'], tf.float32) - 1.0, 0.0)
    arrive_probs = tf.nn.softmax(outputs['arrive_logits'], axis=-1)
    depart_probs = tf.nn.softmax(outputs['depart_logits'], axis=-1)
    arrive_ordinal = tf.cast(time_ordinal_lambda, tf.float32) * masked_ordinal_loss_from_probs(arrive_probs, batch['y_arrive_bin24'], batch['mask_arrive_bin24'])
    depart_ordinal = tf.cast(time_ordinal_lambda, tf.float32) * masked_ordinal_loss_from_probs(depart_probs, batch['y_depart_bin24'], batch['mask_depart_bin24'])
    if time_soft_sigma > 1e-8:
        arrive_soft = build_time_soft_targets(batch['y_arrive_bin24'], batch['mask_arrive_bin24'], model.num_time, time_soft_sigma)
        depart_soft = build_time_soft_targets(batch['y_depart_bin24'], batch['mask_depart_bin24'], model.num_time, time_soft_sigma)
        arrive_loss = masked_soft_ce(outputs['arrive_logits'], arrive_soft, batch['mask_arrive_bin24']) + tf.cast(time_distance_lambda, tf.float32) * masked_huber(arrive_true, arrive_expect, batch['mask_arrive_bin24'], delta=1.0) + arrive_ordinal
        depart_loss = masked_soft_ce(outputs['depart_logits'], depart_soft, batch['mask_depart_bin24']) + tf.cast(time_distance_lambda, tf.float32) * masked_huber(depart_true, depart_expect, batch['mask_depart_bin24'], delta=1.0) + depart_ordinal
    else:
        arrive_loss = masked_sparse_ce(outputs['arrive_logits'], batch['y_arrive_bin24'], batch['mask_arrive_bin24'], class_weights['arrive']) + tf.cast(time_distance_lambda, tf.float32) * masked_huber(arrive_true, arrive_expect, batch['mask_arrive_bin24'], delta=1.0) + arrive_ordinal
        depart_loss = masked_sparse_ce(outputs['depart_logits'], batch['y_depart_bin24'], batch['mask_depart_bin24'], class_weights['depart']) + tf.cast(time_distance_lambda, tf.float32) * masked_huber(depart_true, depart_expect, batch['mask_depart_bin24'], delta=1.0) + depart_ordinal
    if bool(model.use_duration_depart):
        duration_loss = masked_sparse_ce(outputs['duration_logits'], duration_labels, duration_mask, None)
        depart_loss = depart_loss + tf.cast(loss_weights.get('duration', 0.0), tf.float32) * duration_loss
    depart_from_duration_expect = tf.clip_by_value(arrive_expect - duration_expect, 0.0, tf.cast(model.num_time - 1, tf.float32))
    depart_duration_adjust = masked_huber(depart_expect, depart_from_duration_expect, batch['mask_depart_bin24'], delta=1.0)
    next_depart_expect = depart_expect[:, 1:]
    dwell_chain_target = tf.clip_by_value(arrive_expect[:, :-1] + dwell_expect[:, :-1], 0.0, tf.cast(model.num_time - 1, tf.float32))
    dwell_chain_mask = tf.cast((batch['mask_arrive_bin24'][:, :-1] > 0) & (batch['mask_depart_bin24'][:, 1:] > 0) & (batch['seq_mask'][:, 1:] > 0), tf.float32)
    depart_dwell_adjust = masked_huber(next_depart_expect, dwell_chain_target, dwell_chain_mask, delta=1.0)
    losses = {
        'purpose': masked_sparse_ce(outputs['purpose_logits'], batch['y_purpose'], batch['mask_purpose'], class_weights['purpose']),
        'destination': masked_sparse_ce(outputs['destination_logits'], batch['y_dest_zone'], batch['mask_dest_zone'], None),
        'arrive': arrive_loss,
        'mode': masked_sparse_ce(outputs['mode_logits'], batch['y_mode'], batch['mask_mode'], class_weights['mode']),
        'depart': depart_loss,
        'duration': masked_sparse_ce(outputs['duration_logits'], duration_labels, duration_mask, None),
        'dwell': masked_sparse_ce(outputs['dwell_logits'], dwell_labels, dwell_mask, None),
        'depart_adjustment': 0.5 * (depart_duration_adjust + depart_dwell_adjust),
        'mode_family': masked_sparse_ce(outputs['mode_family_logits'], y_mode_family, batch['mask_mode'], None),
        'arrive_coarse': masked_sparse_ce(outputs['arrive_coarse_logits'], y_arrive_coarse, batch['mask_arrive_bin24'], None),
        'depart_coarse': masked_sparse_ce(outputs['depart_coarse_logits'], y_depart_coarse, batch['mask_depart_bin24'], None),
    }
    if 'relation_to_main_logits' in outputs and 'y_relation_to_main' in batch:
        losses['relation_to_main'] = masked_sparse_ce(outputs['relation_to_main_logits'], batch['y_relation_to_main'], batch['mask_relation_to_main'], None)
    if 'secondary_insert_logits' in outputs and 'y_secondary_insert' in batch:
        losses['secondary_insert'] = masked_sparse_ce(outputs['secondary_insert_logits'], batch['y_secondary_insert'], batch['mask_secondary_insert'], None)
    if 'main_step_logits' in outputs and 'y_main_step' in batch:
        main_mask = tf.cast(batch['mask_main'], tf.float32)
        losses['main_step'] = masked_sparse_ce_vector(outputs['main_step_logits'], batch['y_main_step'], main_mask)
        losses['main_purpose'] = masked_sparse_ce_vector(outputs['main_purpose_logits'], batch['y_main_purpose'], main_mask)
        losses['main_destination'] = masked_sparse_ce_vector(outputs['main_destination_logits'], batch['y_main_dest_zone'], main_mask)
        losses['main_arrive'] = masked_sparse_ce_vector(outputs['main_arrive_logits'], batch['y_main_arrive_bin24'], main_mask)
        losses['main_depart'] = masked_sparse_ce_vector(outputs['main_depart_logits'], batch['y_main_depart_bin24'], main_mask)
        losses['main_mode'] = masked_sparse_ce_vector(outputs['main_mode_logits'], batch['y_main_mode'], main_mask)
        losses['main_anchor'] = tf.add_n([
            losses['main_step'],
            losses['main_purpose'],
            losses['main_destination'],
            losses['main_arrive'],
            losses['main_depart'],
            losses['main_mode'],
        ]) / 6.0
    temporal_penalty = temporal_consistency_penalty(outputs, tf.cast(batch['seq_mask'], tf.float32))
    total = tf.add_n([tf.cast(loss_weights[name], tf.float32) * losses[name] for name in ['purpose', 'destination', 'arrive', 'mode', 'depart']])
    total += tf.cast(loss_weights.get('mode_family', 0.0), tf.float32) * losses['mode_family']
    total += tf.cast(loss_weights.get('dwell', 0.0), tf.float32) * losses['dwell']
    total += tf.cast(loss_weights.get('depart_adjustment', 0.0), tf.float32) * losses['depart_adjustment']
    total += tf.cast(loss_weights.get('arrive_coarse', 0.0), tf.float32) * losses['arrive_coarse']
    total += tf.cast(loss_weights.get('depart_coarse', 0.0), tf.float32) * losses['depart_coarse']
    if 'main_anchor' in losses:
        total += tf.cast(loss_weights.get('main_anchor', 0.0), tf.float32) * losses['main_anchor']
        total += tf.cast(loss_weights.get('main_destination_aux', 0.0), tf.float32) * losses['main_destination']
    if 'relation_to_main' in losses:
        total += tf.cast(loss_weights.get('relation_to_main', 0.0), tf.float32) * losses['relation_to_main']
    if 'secondary_insert' in losses:
        total += tf.cast(loss_weights.get('secondary_insert', 0.0), tf.float32) * losses['secondary_insert']
    if 'resource_consistency_penalty' in outputs:
        losses['resource_consistency'] = tf.cast(outputs['resource_consistency_penalty'], tf.float32)
        total += tf.cast(loss_weights.get('resource_consistency', 0.0), tf.float32) * losses['resource_consistency']
    total += tf.cast(temporal_lambda, tf.float32) * temporal_penalty
    losses['temporal_penalty'] = temporal_penalty
    losses['total'] = total
    return total, losses, outputs


@tf.function
def train_step(model: TourInterpretableGraphACTNN, optimizer: tf.keras.optimizers.Optimizer, batch: dict[str, tf.Tensor], class_weights: dict[str, tf.Tensor], loss_weights: dict[str, float], temporal_lambda: float, time_soft_sigma: float, time_distance_lambda: float, time_ordinal_lambda: float) -> dict[str, tf.Tensor]:
    with tf.GradientTape() as tape:
        total_loss, loss_parts, _ = compute_total_loss(model, batch, class_weights, loss_weights, temporal_lambda, time_soft_sigma, time_distance_lambda, time_ordinal_lambda, training=True)
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_parts


def prepare_schedule_sampling_batch(model: TourInterpretableGraphACTNN, batch: dict[str, tf.Tensor], ss_prob: float, zone_district_lookup: np.ndarray, ss_time_boost: float, ss_late_boost: float) -> dict[str, tf.Tensor]:
    if ss_prob <= 1e-8 and ss_time_boost <= 1e-8 and ss_late_boost <= 1e-8:
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
    pred_arrive = tf.argmax(outputs['arrive_logits'], axis=-1, output_type=tf.int32).numpy()
    pred_mode = tf.argmax(outputs['mode_logits'], axis=-1, output_type=tf.int32).numpy()
    pred_depart = tf.argmax(outputs['depart_logits'], axis=-1, output_type=tf.int32).numpy()
    seq_mask = batch_np['seq_mask'] > 0
    batch_size, max_steps, _ = batch_np['step_cat'].shape
    num_transitions = max(max_steps - 1, 1)
    step_scale = np.arange(num_transitions, dtype=np.float32) / max(num_transitions - 1, 1)
    state_probs = np.clip(ss_prob + ss_late_boost * step_scale, 0.0, 0.95)
    time_probs = np.clip(ss_prob + ss_time_boost + ss_late_boost * step_scale, 0.0, 0.98)
    use_pred_state_matrix = np.random.rand(batch_size, num_transitions) < state_probs[np.newaxis, :]
    use_pred_time_matrix = np.random.rand(batch_size, num_transitions) < time_probs[np.newaxis, :]
    for step_idx in range(max_steps - 1):
        next_mask = seq_mask[:, step_idx + 1]
        use_pred_state = use_pred_state_matrix[:, step_idx] & next_mask
        use_pred_time = use_pred_time_matrix[:, step_idx] & next_mask
        if not np.any(use_pred_state) and not np.any(use_pred_time):
            continue
        time_denom = float(max(model.num_time - 2, 1))
        if np.any(use_pred_state):
            origin_next = pred_dest[use_pred_state, step_idx]
            batch_np['step_cat'][use_pred_state, step_idx + 1, 0] = origin_next
            batch_np['step_cat'][use_pred_state, step_idx + 1, 1] = pred_purpose[use_pred_state, step_idx]
            batch_np['step_cat'][use_pred_state, step_idx + 1, 2] = pred_mode[use_pred_state, step_idx]
            batch_np['step_cat'][use_pred_state, step_idx + 1, 3] = zone_district_lookup[origin_next]
            batch_np['step_num'][use_pred_state, step_idx + 1, 3] = (origin_next == batch_np['static_cat'][use_pred_state, 5]).astype(np.float32)
        if np.any(use_pred_time):
            batch_np['step_num'][use_pred_time, step_idx + 1, 1] = np.maximum(pred_depart[use_pred_time, step_idx] - 1, 0) / time_denom
            batch_np['step_num'][use_pred_time, step_idx + 1, 2] = np.maximum(pred_arrive[use_pred_time, step_idx] - 1, 0) / time_denom
    return {key: tf.convert_to_tensor(value) for key, value in batch_np.items()}


def run_epoch(model: TourInterpretableGraphACTNN, optimizer: tf.keras.optimizers.Optimizer, dataset: tf.data.Dataset, class_weights: dict[str, tf.Tensor], loss_weights: dict[str, float], temporal_lambda: float, time_soft_sigma: float, time_distance_lambda: float, time_ordinal_lambda: float, ss_prob: float, zone_district_lookup: np.ndarray, ss_time_boost: float, ss_late_boost: float) -> dict[str, float]:
    sums: dict[str, float] = {}
    steps = 0
    for batch in dataset:
        batch_for_train = prepare_schedule_sampling_batch(model, batch, ss_prob, zone_district_lookup, ss_time_boost, ss_late_boost)
        loss_parts = train_step(model, optimizer, batch_for_train, class_weights, loss_weights, temporal_lambda, time_soft_sigma, time_distance_lambda, time_ordinal_lambda)
        for key, value in loss_parts.items():
            sums[key] = sums.get(key, 0.0) + float(value.numpy())
        steps += 1
    return {k: v / max(steps, 1) for k, v in sums.items()}


def evaluate_loss(model: TourInterpretableGraphACTNN, dataset: tf.data.Dataset, class_weights: dict[str, tf.Tensor], loss_weights: dict[str, float], temporal_lambda: float, time_soft_sigma: float, time_distance_lambda: float, time_ordinal_lambda: float) -> dict[str, float]:
    sums: dict[str, float] = {}
    steps = 0
    for batch in dataset:
        _, loss_parts, _ = compute_total_loss(model, batch, class_weights, loss_weights, temporal_lambda, time_soft_sigma, time_distance_lambda, time_ordinal_lambda, training=False)
        for key, value in loss_parts.items():
            sums[key] = sums.get(key, 0.0) + float(value.numpy())
        steps += 1
    return {k: v / max(steps, 1) for k, v in sums.items()}


def forward_batches(model: TourInterpretableGraphACTNN, arrays: dict[str, np.ndarray], batch_size: int, case_person_limit: int = 30, autoregressive: bool = False, use_observed_origin: bool = False, zone_district_lookup: np.ndarray | None = None) -> dict[str, Any]:
    n = len(arrays['person_id'])
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in ['purpose', 'destination', 'arrive', 'mode', 'depart']}
    dest_hits: dict[int, list[np.ndarray]] = {k: [] for k in TOPK_VALUES}
    case_rows: list[dict[str, Any]] = []
    destination_topk_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    captured_case_people: set[int] = set()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_np = {key: value[start:end] for key, value in arrays.items()}
        need_attention = len(captured_case_people) < min(case_person_limit, 5)
        if not autoregressive:
            batch = {key: tf.convert_to_tensor(value) for key, value in batch_np.items()}
            outputs = model(split_inputs(batch), training=False, return_attention=need_attention)
            purpose_pred = tf.argmax(outputs['purpose_logits'], axis=-1, output_type=tf.int32).numpy()
            dest_pred = tf.argmax(outputs['destination_logits'], axis=-1, output_type=tf.int32).numpy()
            arrive_pred = tf.argmax(outputs['arrive_logits'], axis=-1, output_type=tf.int32).numpy()
            mode_pred = tf.argmax(outputs['mode_logits'], axis=-1, output_type=tf.int32).numpy()
            depart_pred = tf.argmax(outputs['depart_logits'], axis=-1, output_type=tf.int32).numpy()
            topk_indices = tf.math.top_k(outputs['destination_logits'], k=max(TOPK_VALUES)).indices.numpy()
            final_attention_outputs = outputs
        else:
            step_cat_ar = batch_np['step_cat'].copy()
            step_num_ar = batch_np['step_num'].copy()
            progress_mask = np.zeros_like(batch_np['seq_mask'])
            batch_size_local, max_steps, _ = step_cat_ar.shape
            home_zone_idx_batch = batch_np['static_cat'][:, 5].copy()
            current_origin = home_zone_idx_batch.copy()
            purpose_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            dest_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            arrive_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            mode_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            depart_pred = np.zeros((batch_size_local, max_steps), dtype=np.int32)
            topk_indices = np.zeros((batch_size_local, max_steps, max(TOPK_VALUES)), dtype=np.int32)

            for step_idx in range(max_steps):
                active_mask = batch_np['seq_mask'][:, step_idx] > 0
                if not np.any(active_mask):
                    continue
                progress_mask[active_mask, step_idx] = 1.0
                if use_observed_origin:
                    step_cat_ar[active_mask, step_idx, 0] = batch_np['step_cat'][active_mask, step_idx, 0]
                    if zone_district_lookup is not None:
                        step_cat_ar[active_mask, step_idx, 3] = zone_district_lookup[step_cat_ar[active_mask, step_idx, 0]]
                else:
                    step_cat_ar[active_mask, step_idx, 0] = current_origin[active_mask]
                    if zone_district_lookup is not None:
                        step_cat_ar[active_mask, step_idx, 3] = zone_district_lookup[current_origin[active_mask]]
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
                    'seq_mask': tf.convert_to_tensor(progress_mask),
                }
                outputs_step = model(batch_inputs, training=False, return_attention=False)
                purpose_step = tf.argmax(outputs_step['purpose_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                dest_step = tf.argmax(outputs_step['destination_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                arrive_step = tf.argmax(outputs_step['arrive_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                mode_step = tf.argmax(outputs_step['mode_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                depart_step = tf.argmax(outputs_step['depart_logits'][:, step_idx, :], axis=-1, output_type=tf.int32).numpy()
                topk_step = tf.math.top_k(outputs_step['destination_logits'][:, step_idx, :], k=max(TOPK_VALUES)).indices.numpy()

                purpose_pred[:, step_idx] = purpose_step
                dest_pred[:, step_idx] = dest_step
                arrive_pred[:, step_idx] = arrive_step
                mode_pred[:, step_idx] = mode_step
                depart_pred[:, step_idx] = depart_step
                topk_indices[:, step_idx, :] = topk_step

                next_idx = step_idx + 1
                if next_idx < max_steps:
                    next_active = batch_np['seq_mask'][:, next_idx] > 0
                    if np.any(next_active):
                        if not use_observed_origin:
                            current_origin[next_active] = dest_step[next_active]
                            if zone_district_lookup is not None:
                                step_cat_ar[next_active, next_idx, 3] = zone_district_lookup[current_origin[next_active]]
                            step_num_ar[next_active, next_idx, 3] = (current_origin[next_active] == home_zone_idx_batch[next_active]).astype(np.float32)
                        step_cat_ar[next_active, next_idx, 1] = purpose_step[next_active]
                        step_cat_ar[next_active, next_idx, 2] = mode_step[next_active]
                        time_denom = float(max(model.num_time - 2, 1))
                        step_num_ar[next_active, next_idx, 1] = np.maximum(depart_step[next_active] - 1, 0) / time_denom
                        step_num_ar[next_active, next_idx, 2] = np.maximum(arrive_step[next_active] - 1, 0) / time_denom

            final_inputs = {
                'static_cat': tf.convert_to_tensor(batch_np['static_cat']),
                'static_num': tf.convert_to_tensor(batch_np['static_num']),
                'step_cat': tf.convert_to_tensor(step_cat_ar),
                'step_num': tf.convert_to_tensor(step_num_ar),
                'seq_mask': tf.convert_to_tensor(progress_mask),
            }
            final_attention_outputs = model(final_inputs, training=False, return_attention=need_attention)

        predictions['purpose'].append(purpose_pred)
        predictions['destination'].append(dest_pred)
        predictions['arrive'].append(arrive_pred)
        predictions['mode'].append(mode_pred)
        predictions['depart'].append(depart_pred)
        true_dest = batch_np['y_dest_zone']
        for k in TOPK_VALUES:
            hits = np.any(topk_indices[:, :, :k] == true_dest[:, :, None], axis=-1).astype(np.float32)
            dest_hits[k].append(hits)

        if len(captured_case_people) < case_person_limit:
            batch_persons = batch_np['person_id']
            for local_idx, person_id in enumerate(batch_persons):
                if int(person_id) in captured_case_people:
                    continue
                captured_case_people.add(int(person_id))
                seq_len = int(batch_np['seq_len'][local_idx])
                for step_idx in range(seq_len):
                    destination_topk_rows.append({
                        'person_id': int(person_id),
                        'household_id': int(batch_np['household_id'][local_idx]),
                        'step_index': int(step_idx),
                        'inference_mode': 'fixed_length_autoregressive' if autoregressive else 'conditional_teacher_forcing',
                        'true_dest_zone_idx': int(batch_np['y_dest_zone'][local_idx, step_idx]),
                        'pred_dest_zone_idx': int(dest_pred[local_idx, step_idx]),
                        'top5_zone_idx': '|'.join(map(str, topk_indices[local_idx, step_idx, :5].tolist())),
                    })
                    case_rows.append({
                        'person_id': int(person_id),
                        'household_id': int(batch_np['household_id'][local_idx]),
                        'step_index': int(step_idx),
                        'inference_mode': 'fixed_length_autoregressive' if autoregressive else 'conditional_teacher_forcing',
                        'true_purpose': int(batch_np['y_purpose'][local_idx, step_idx]),
                        'pred_purpose': int(purpose_pred[local_idx, step_idx]),
                        'true_dest_zone': int(batch_np['y_dest_zone'][local_idx, step_idx]),
                        'pred_dest_zone': int(dest_pred[local_idx, step_idx]),
                        'true_arrive_bin24': int(batch_np['y_arrive_bin24'][local_idx, step_idx]),
                        'pred_arrive_bin24': int(arrive_pred[local_idx, step_idx]),
                        'true_mode': int(batch_np['y_mode'][local_idx, step_idx]),
                        'pred_mode': int(mode_pred[local_idx, step_idx]),
                        'true_depart_bin24': int(batch_np['y_depart_bin24'][local_idx, step_idx]),
                        'pred_depart_bin24': int(depart_pred[local_idx, step_idx]),
                    })
                if len(captured_case_people) >= case_person_limit:
                    break

        if need_attention and 'attention_scores' in final_attention_outputs:
            scores = final_attention_outputs['attention_scores'].numpy()
            batch_persons = batch_np['person_id']
            max_cases = min(len(batch_persons), max(0, 5 - len(attention_rows)))
            for local_idx in range(max_cases):
                person_id = int(batch_persons[local_idx])
                seq_len = int(batch_np['seq_len'][local_idx])
                mean_scores = scores[local_idx].mean(axis=0)
                for from_step in range(seq_len):
                    for to_step in range(seq_len):
                        attention_rows.append({'person_id': person_id, 'from_step': int(from_step), 'to_step': int(to_step), 'attention_score': float(mean_scores[from_step, to_step])})




    pred_arrays = {key: np.concatenate(value, axis=0) for key, value in predictions.items()}
    topk_arrays = {k: np.concatenate(value, axis=0) for k, value in dest_hits.items()}
    return {
        'predictions': pred_arrays,
        'destination_topk_hits': topk_arrays,
        'case_rows': case_rows,
        'destination_topk_rows': destination_topk_rows,
        'attention_rows': attention_rows,
        'inference_mode': 'fixed_length_autoregressive' if autoregressive else 'conditional_teacher_forcing',
    }

def compute_main_activity_metrics(arrays: dict[str, np.ndarray], inference: dict[str, Any], purpose_vocab: dict[str, int], split_name: str) -> pd.DataFrame:
    pred = inference['predictions']
    pred_arrays = {
        'seq_mask': arrays['seq_mask'].copy(),
        'mask_arrive_bin24': arrays['mask_arrive_bin24'].copy(),
        'mask_depart_bin24': arrays['mask_depart_bin24'].copy(),
        'y_purpose': pred['purpose'].astype(np.int32).copy(),
        'y_dest_zone': pred['destination'].astype(np.int32).copy(),
        'y_arrive_bin24': pred['arrive'].astype(np.int32).copy(),
        'y_depart_bin24': pred['depart'].astype(np.int32).copy(),
        'y_mode': pred['mode'].astype(np.int32).copy(),
    }
    pred_main = augment_main_targets(pred_arrays, purpose_vocab)
    mask = arrays['mask_main'] > 0
    n_main = int(mask.sum())
    if n_main == 0:
        return pd.DataFrame([{
            'split': split_name,
            'n_main_samples': 0,
            'main_step_acc': float('nan'),
            'main_purpose_acc': float('nan'),
            'main_destination_top1': float('nan'),
            'main_arrive_acc': float('nan'),
            'main_depart_acc': float('nan'),
            'main_mode_acc': float('nan'),
            'main_task_mean_acc': float('nan'),
            'main_exact_all_tasks_and_step_acc': float('nan'),
        }])
    main_step_acc = float((pred_main['y_main_step'][mask] == arrays['y_main_step'][mask]).mean())
    main_purpose_acc = float((pred_main['y_main_purpose'][mask] == arrays['y_main_purpose'][mask]).mean())
    main_destination_top1 = float((pred_main['y_main_dest_zone'][mask] == arrays['y_main_dest_zone'][mask]).mean())
    main_arrive_acc = float((pred_main['y_main_arrive_bin24'][mask] == arrays['y_main_arrive_bin24'][mask]).mean())
    main_depart_acc = float((pred_main['y_main_depart_bin24'][mask] == arrays['y_main_depart_bin24'][mask]).mean())
    main_mode_acc = float((pred_main['y_main_mode'][mask] == arrays['y_main_mode'][mask]).mean())
    main_exact = (
        (pred_main['y_main_step'][mask] == arrays['y_main_step'][mask])
        & (pred_main['y_main_purpose'][mask] == arrays['y_main_purpose'][mask])
        & (pred_main['y_main_dest_zone'][mask] == arrays['y_main_dest_zone'][mask])
        & (pred_main['y_main_arrive_bin24'][mask] == arrays['y_main_arrive_bin24'][mask])
        & (pred_main['y_main_depart_bin24'][mask] == arrays['y_main_depart_bin24'][mask])
        & (pred_main['y_main_mode'][mask] == arrays['y_main_mode'][mask])
    )
    return pd.DataFrame([{
        'split': split_name,
        'n_main_samples': n_main,
        'main_step_acc': main_step_acc,
        'main_purpose_acc': main_purpose_acc,
        'main_destination_top1': main_destination_top1,
        'main_arrive_acc': main_arrive_acc,
        'main_depart_acc': main_depart_acc,
        'main_mode_acc': main_mode_acc,
        'main_task_mean_acc': float(np.mean([main_purpose_acc, main_destination_top1, main_arrive_acc, main_depart_acc, main_mode_acc])),
        'main_exact_all_tasks_and_step_acc': float(main_exact.mean()),
    }])

def summarize_metrics(split_name: str, arrays: dict[str, np.ndarray], inference: dict[str, Any], time_bin_size_hours: int, mode_eval_labels: list[int]) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = inference['predictions']
    topk_hits = inference['destination_topk_hits']
    step_exact_mask = ((arrays['mask_purpose'] == 0) | (arrays['y_purpose'] == pred['purpose'])) & ((arrays['mask_dest_zone'] == 0) | (arrays['y_dest_zone'] == pred['destination'])) & ((arrays['mask_arrive_bin24'] == 0) | (arrays['y_arrive_bin24'] == pred['arrive'])) & ((arrays['mask_mode'] == 0) | (arrays['y_mode'] == pred['mode'])) & ((arrays['mask_depart_bin24'] == 0) | (arrays['y_depart_bin24'] == pred['depart']))
    step_valid_mask = (arrays['seq_mask'] > 0)
    metrics = {
        'split': split_name,
        'inference_mode': inference['inference_mode'],
        'purpose_acc': masked_accuracy(arrays['y_purpose'], pred['purpose'], arrays['mask_purpose']),
        'purpose_macro_f1': decode_metrics_f1(arrays['y_purpose'][arrays['mask_purpose'] > 0], pred['purpose'][arrays['mask_purpose'] > 0], [1, 2, 3, 4, 5, 6]),
        'destination_top1': topk_accuracy(topk_hits[1], arrays['mask_dest_zone']),
        'destination_top5': topk_accuracy(topk_hits[5], arrays['mask_dest_zone']),
        'destination_top10': topk_accuracy(topk_hits[10], arrays['mask_dest_zone']),
        'destination_top20': topk_accuracy(topk_hits[20], arrays['mask_dest_zone']),
        'arrive_acc': masked_accuracy(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
        'depart_acc': masked_accuracy(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
        'arrive_within1_acc': within_one_bin_accuracy(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
        'depart_within1_acc': within_one_bin_accuracy(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
        'arrive_mae_hours': masked_mae(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']) * time_bin_size_hours,
        'depart_mae_hours': masked_mae(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']) * time_bin_size_hours,
        'mode_acc': masked_accuracy(arrays['y_mode'], pred['mode'], arrays['mask_mode']),
        'mode_macro_f1': decode_metrics_f1(arrays['y_mode'][arrays['mask_mode'] > 0], pred['mode'][arrays['mask_mode'] > 0], mode_eval_labels),
        'overall_mean_acc': float(np.mean([
            masked_accuracy(arrays['y_purpose'], pred['purpose'], arrays['mask_purpose']),
            topk_accuracy(topk_hits[1], arrays['mask_dest_zone']),
            masked_accuracy(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
            masked_accuracy(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
            masked_accuracy(arrays['y_mode'], pred['mode'], arrays['mask_mode']),
        ])),
        'time_mean_acc': float(np.mean([
            masked_accuracy(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
            masked_accuracy(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
        ])),
        'time_within1_mean_acc': float(np.mean([
            within_one_bin_accuracy(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
            within_one_bin_accuracy(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
        ])),
        'time_score': float(np.mean([
            masked_accuracy(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']),
            masked_accuracy(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']),
        ])) - 0.1 * (
            masked_mae(arrays['y_arrive_bin24'], pred['arrive'], arrays['mask_arrive_bin24']) * time_bin_size_hours +
            masked_mae(arrays['y_depart_bin24'], pred['depart'], arrays['mask_depart_bin24']) * time_bin_size_hours
        ),
        'step_exact_all_tasks_acc': float(step_exact_mask[step_valid_mask].mean()) if step_valid_mask.sum() else float('nan'),
    }
    by_step_rows = []
    for step_idx in range(arrays['seq_mask'].shape[1]):
        by_step_rows.append({
            'split': split_name,
            'step_index': step_idx,
            'n_samples': int(arrays['seq_mask'][:, step_idx].sum()),
            'purpose_acc': masked_accuracy(arrays['y_purpose'][:, step_idx], pred['purpose'][:, step_idx], arrays['mask_purpose'][:, step_idx]),
            'destination_top1': topk_accuracy(topk_hits[1][:, step_idx], arrays['mask_dest_zone'][:, step_idx]),
            'destination_top10': topk_accuracy(topk_hits[10][:, step_idx], arrays['mask_dest_zone'][:, step_idx]),
            'arrive_acc': masked_accuracy(arrays['y_arrive_bin24'][:, step_idx], pred['arrive'][:, step_idx], arrays['mask_arrive_bin24'][:, step_idx]),
            'depart_acc': masked_accuracy(arrays['y_depart_bin24'][:, step_idx], pred['depart'][:, step_idx], arrays['mask_depart_bin24'][:, step_idx]),
            'arrive_within1_acc': within_one_bin_accuracy(arrays['y_arrive_bin24'][:, step_idx], pred['arrive'][:, step_idx], arrays['mask_arrive_bin24'][:, step_idx]),
            'depart_within1_acc': within_one_bin_accuracy(arrays['y_depart_bin24'][:, step_idx], pred['depart'][:, step_idx], arrays['mask_depart_bin24'][:, step_idx]),
            'mode_acc': masked_accuracy(arrays['y_mode'][:, step_idx], pred['mode'][:, step_idx], arrays['mask_mode'][:, step_idx]),
        })
    by_step_df = pd.DataFrame(by_step_rows)
    topk_rows = []
    for k in TOPK_VALUES:
        topk_rows.append({'split': split_name, 'k': k, 'step_index': -1, 'topk_accuracy': topk_accuracy(topk_hits[k], arrays['mask_dest_zone'])})
        for step_idx in range(arrays['seq_mask'].shape[1]):
            topk_rows.append({'split': split_name, 'k': k, 'step_index': step_idx, 'topk_accuracy': topk_accuracy(topk_hits[k][:, step_idx], arrays['mask_dest_zone'][:, step_idx])})
    topk_df = pd.DataFrame(topk_rows)
    chain_exact = np.all(
        ((arrays['mask_purpose'] == 0) | (arrays['y_purpose'] == pred['purpose']))
        & ((arrays['mask_dest_zone'] == 0) | (arrays['y_dest_zone'] == pred['destination']))
        & ((arrays['mask_arrive_bin24'] == 0) | (arrays['y_arrive_bin24'] == pred['arrive']))
        & ((arrays['mask_mode'] == 0) | (arrays['y_mode'] == pred['mode']))
        & ((arrays['mask_depart_bin24'] == 0) | (arrays['y_depart_bin24'] == pred['depart'])),
        axis=1,
    )
    chain_df = pd.DataFrame([{'chain_exact_all_tasks_rate': float(chain_exact.mean()), 'mean_seq_len': float(arrays['seq_len'].mean())}])
    same_step = (arrays['mask_depart_bin24'] > 0) & (arrays['mask_arrive_bin24'] > 0)
    adjacent = (arrays['mask_depart_bin24'][:, 1:] > 0) & (arrays['mask_arrive_bin24'][:, :-1] > 0)
    same_step_violation = ((pred['depart'] > pred['arrive']) & same_step).astype(np.float32)
    adjacent_violation = ((pred['depart'][:, 1:] < pred['arrive'][:, :-1]) & adjacent).astype(np.float32)
    behavior_df = pd.DataFrame([{
        'split': split_name,
        'same_step_time_violation_rate': float(same_step_violation[same_step].mean()) if same_step.sum() else float('nan'),
        'adjacent_time_violation_rate': float(adjacent_violation[adjacent].mean()) if adjacent.sum() else float('nan'),
    }])
    return metrics, by_step_df, topk_df, chain_df, behavior_df


def write_png_figures(run_dir: Path, history_df: pd.DataFrame, results_df: pd.DataFrame, figure_frames: dict[str, pd.DataFrame]) -> None:
    fig_dir = run_dir / 'figure_data'
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not history_df.empty:
        plt.figure(figsize=(8, 5))
        if 'train_total' in history_df.columns:
            plt.plot(history_df['epoch'], history_df['train_total'], label='train_total', marker='o')
        if 'valid_total' in history_df.columns:
            plt.plot(history_df['epoch'], history_df['valid_total'], label='valid_total', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Convergence Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'convergence_loss.png', dpi=180)
        plt.close()

        metric_cols = [c for c in history_df.columns if c.startswith('valid_metric_')]
        if metric_cols:
            plt.figure(figsize=(8, 5))
            for col in metric_cols:
                plt.plot(history_df['epoch'], history_df[col], label=col.replace('valid_metric_', ''), marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Metric')
            plt.title('Validation Metric Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / 'validation_metric_curve.png', dpi=180)
            plt.close()

    test_df = results_df[results_df['split'] == 'test']
    if not test_df.empty:
        row = test_df.iloc[0]
        names = ['purpose_acc', 'destination_top1', 'arrive_acc', 'depart_acc', 'mode_acc']
        vals = [float(row[n]) for n in names]
        plt.figure(figsize=(8, 5))
        plt.bar(names, vals)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('Test Task Accuracy')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(fig_dir / 'test_task_accuracy.png', dpi=180)
        plt.close()

        time_names = ['arrive_acc', 'depart_acc', 'arrive_within1_acc', 'depart_within1_acc']
        time_vals = [float(row[n]) for n in time_names if n in row.index]
        time_labels = [n for n in time_names if n in row.index]
        if time_vals:
            plt.figure(figsize=(8, 5))
            plt.bar(time_labels, time_vals)
            plt.ylim(0, 1)
            plt.ylabel('Accuracy')
            plt.title('Test Time Accuracy')
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(fig_dir / 'test_time_accuracy.png', dpi=180)
            plt.close()

    step_key = 'test_metrics_by_step'
    if step_key in figure_frames and not figure_frames[step_key].empty:
        df = figure_frames[step_key]
        plt.figure(figsize=(9, 5))
        for col in ['purpose_acc', 'destination_top1', 'arrive_acc', 'depart_acc', 'mode_acc']:
            if col in df.columns:
                plt.plot(df['step_index'], df[col], marker='o', label=col)
        plt.xlabel('Step Index')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Step')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'accuracy_by_step.png', dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        for col in ['arrive_acc', 'depart_acc', 'arrive_within1_acc', 'depart_within1_acc']:
            if col in df.columns:
                plt.plot(df['step_index'], df[col], marker='o', label=col)
        plt.xlabel('Step Index')
        plt.ylabel('Accuracy')
        plt.title('Time Accuracy by Step')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'time_accuracy_by_step.png', dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the tour-based hierarchical ACT-NN model.')
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
    parser.add_argument('--temporal-lambda', type=float, default=0.10)
    parser.add_argument('--destination-loss-weight', type=float, default=1.25)
    parser.add_argument('--arrive-loss-weight', type=float, default=1.1)
    parser.add_argument('--depart-loss-weight', type=float, default=1.0)
    parser.add_argument('--mode-loss-weight', type=float, default=2.0)
    parser.add_argument('--mode-family-loss-weight', type=float, default=0.8)
    parser.add_argument('--arrive-coarse-loss-weight', type=float, default=0.6)
    parser.add_argument('--depart-coarse-loss-weight', type=float, default=0.6)
    parser.add_argument('--time-bin-size-hours', type=int, default=3)
    parser.add_argument('--mode-schema', type=str, default='fine7', choices=['fine7', 'coarse4', 'standard3'])
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subset-train', type=int, default=None)
    parser.add_argument('--subset-valid', type=int, default=None)
    parser.add_argument('--subset-test', type=int, default=None)
    parser.add_argument('--schedule-sampling-max', type=float, default=0.35)
    parser.add_argument('--schedule-sampling-warmup', type=int, default=4)
    parser.add_argument('--schedule-sampling-time-boost', type=float, default=0.15)
    parser.add_argument('--schedule-sampling-late-boost', type=float, default=0.10)
    parser.add_argument('--inference-origin', type=str, default='predicted', choices=['predicted', 'observed'])
    parser.add_argument('--evaluation-mode', type=str, default='half_autoregressive', choices=['conditional', 'half_autoregressive'])
    parser.add_argument('--candidate-knn-k', type=int, default=64)
    parser.add_argument('--candidate-penalty', type=float, default=-2.5)
    parser.add_argument('--destination-context-k', type=int, default=8)
    parser.add_argument('--destination-context-topk-alpha', type=float, default=0.7)
    parser.add_argument('--checkpoint-metric', type=str, default='overall_mean_acc', choices=['loss', 'overall_mean_acc', 'destination_top1', 'mode_acc', 'arrive_acc', 'depart_acc', 'time_mean_acc', 'time_score'])
    parser.add_argument('--time-soft-sigma', type=float, default=0.0)
    parser.add_argument('--time-distance-loss-weight', type=float, default=0.0)
    parser.add_argument('--time-ordinal-loss-weight', type=float, default=0.0)
    parser.add_argument('--use-duration-depart', action='store_true')
    parser.add_argument('--duration-loss-weight', type=float, default=0.4)
    parser.add_argument('--dwell-loss-weight', type=float, default=0.35)
    parser.add_argument('--depart-adjustment-loss-weight', type=float, default=0.2)
    parser.add_argument('--duration-depart-blend', type=float, default=0.6)
    parser.add_argument('--use-depart-adjustment', action='store_true')
    parser.add_argument('--depart-dwell-bias-scale', type=float, default=0.25)
    parser.add_argument('--predict-depart-first', action='store_true')
    parser.add_argument('--use-main-anchor-conditioning', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--main-anchor-loss-weight', type=float, default=0.12)
    parser.add_argument('--main-destination-aux-loss-weight', type=float, default=0.0)
    parser.add_argument('--relation-to-main-loss-weight', type=float, default=0.20)
    parser.add_argument('--secondary-insert-loss-weight', type=float, default=0.25)
    parser.add_argument('--resource-consistency-loss-weight', type=float, default=0.20)
    parser.add_argument('--main-anchor-conditioning-warmup', type=int, default=2)
    parser.add_argument('--main-anchor-conditioning-ramp', type=int, default=2)
    parser.add_argument('--main-anchor-loss-warmup', type=int, default=1)
    parser.add_argument('--main-anchor-loss-ramp', type=int, default=2)
    parser.add_argument('--relation-loss-warmup', type=int, default=1)
    parser.add_argument('--relation-loss-ramp', type=int, default=2)
    parser.add_argument('--secondary-insert-loss-warmup', type=int, default=2)
    parser.add_argument('--secondary-insert-loss-ramp', type=int, default=2)
    parser.add_argument('--use-tour-resource-state', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-relation-to-main-conditioning', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-secondary-local-destination', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-interpretable-modules', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resource-conditioning-warmup', type=int, default=3)
    parser.add_argument('--resource-conditioning-ramp', type=int, default=2)
    parser.add_argument('--resource-loss-warmup', type=int, default=3)
    parser.add_argument('--resource-loss-ramp', type=int, default=2)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime('actnn_tour_graph_%Y%m%d_%H%M%S')
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
        'temporal_lambda': args.temporal_lambda,
        'destination_loss_weight': args.destination_loss_weight,
        'arrive_loss_weight': args.arrive_loss_weight,
        'depart_loss_weight': args.depart_loss_weight,
        'mode_loss_weight': args.mode_loss_weight,
        'mode_family_loss_weight': args.mode_family_loss_weight,
        'arrive_coarse_loss_weight': args.arrive_coarse_loss_weight,
        'depart_coarse_loss_weight': args.depart_coarse_loss_weight,
        'time_bin_size_hours': args.time_bin_size_hours,
        'mode_schema': args.mode_schema,
        'num_time_classes': num_time_classes,
        'patience': args.patience,
        'seed': args.seed,
        'subset_train': args.subset_train,
        'subset_valid': args.subset_valid,
        'subset_test': args.subset_test,
        'schedule_sampling_max': args.schedule_sampling_max,
        'schedule_sampling_warmup': args.schedule_sampling_warmup,
        'schedule_sampling_time_boost': args.schedule_sampling_time_boost,
        'schedule_sampling_late_boost': args.schedule_sampling_late_boost,
        'inference_origin': args.inference_origin,
        'evaluation_mode': args.evaluation_mode,
        'candidate_knn_k': args.candidate_knn_k,
        'candidate_penalty': args.candidate_penalty,
        'destination_context_k': args.destination_context_k,
        'destination_context_topk_alpha': args.destination_context_topk_alpha,
        'checkpoint_metric': args.checkpoint_metric,
        'time_soft_sigma': args.time_soft_sigma,
        'time_distance_loss_weight': args.time_distance_loss_weight,
        'time_ordinal_loss_weight': args.time_ordinal_loss_weight,
        'use_duration_depart': args.use_duration_depart,
        'duration_loss_weight': args.duration_loss_weight,
        'dwell_loss_weight': args.dwell_loss_weight,
        'depart_adjustment_loss_weight': args.depart_adjustment_loss_weight,
        'duration_depart_blend': args.duration_depart_blend,
        'use_depart_adjustment': args.use_depart_adjustment,
        'depart_dwell_bias_scale': args.depart_dwell_bias_scale,
        'predict_depart_first': args.predict_depart_first,
        'use_main_anchor_conditioning': args.use_main_anchor_conditioning,
        'use_relation_to_main_conditioning': args.use_relation_to_main_conditioning,
        'use_secondary_local_destination': args.use_secondary_local_destination,
        'use_interpretable_modules': args.use_interpretable_modules,
        'main_anchor_loss_weight': args.main_anchor_loss_weight,
        'main_destination_aux_loss_weight': args.main_destination_aux_loss_weight,
        'main_anchor_conditioning_warmup': args.main_anchor_conditioning_warmup,
        'main_anchor_conditioning_ramp': args.main_anchor_conditioning_ramp,
        'main_anchor_loss_warmup': args.main_anchor_loss_warmup,
        'main_anchor_loss_ramp': args.main_anchor_loss_ramp,
        'relation_loss_warmup': args.relation_loss_warmup,
        'relation_loss_ramp': args.relation_loss_ramp,
        'secondary_insert_loss_warmup': args.secondary_insert_loss_warmup,
        'secondary_insert_loss_ramp': args.secondary_insert_loss_ramp,
        'use_tour_resource_state': args.use_tour_resource_state,
        'num_relation': 6,
        'resource_conditioning_warmup': args.resource_conditioning_warmup,
        'resource_conditioning_ramp': args.resource_conditioning_ramp,
        'resource_loss_warmup': args.resource_loss_warmup,
        'resource_loss_ramp': args.resource_loss_ramp,
        'max_steps': dataset_summary['max_steps'],
        'home_purpose_id': int(vocab['purpose_target_vocab']['Home']),
        'work_purpose_id': int(vocab['purpose_target_vocab']['Work']),
        'bos_purpose_id': int(vocab['bos_purpose_id']),
        'bos_mode_id': int(vocab['bos_mode_id']),
    }
    mode_vocab, mode_old_to_new, new_bos_mode_id = build_mode_schema(vocab['mode_target_vocab'], args.mode_schema, int(vocab['bos_mode_id']))
    vocab = dict(vocab)
    vocab['mode_target_vocab'] = mode_vocab
    vocab['bos_mode_id'] = int(new_bos_mode_id)
    mode_eval_labels = sorted(v for k, v in mode_vocab.items() if v > 0)
    config.update({
        'walk_mode_id': int(mode_vocab.get('Walk', 0)),
        'bike_mode_id': int(mode_vocab.get('BikeEbike', 0)),
        'bus_mode_id': int(mode_vocab.get('Bus', 0)),
        'metro_mode_id': int(mode_vocab.get('Metro', 0)),
        'taxi_mode_id': int(mode_vocab.get('TaxiRidehail', 0)),
        'car_mode_id': int(mode_vocab.get('CarMotor', 0)),
    })
    (run_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    train_arrays = remap_mode_arrays(remap_time_arrays(load_split_arrays('train', args.subset_train), args.time_bin_size_hours), mode_old_to_new, new_bos_mode_id)
    valid_arrays = remap_mode_arrays(remap_time_arrays(load_split_arrays('valid', args.subset_valid), args.time_bin_size_hours), mode_old_to_new, new_bos_mode_id)
    test_arrays = remap_mode_arrays(remap_time_arrays(load_split_arrays('test', args.subset_test), args.time_bin_size_hours), mode_old_to_new, new_bos_mode_id)

    train_arrays = augment_secondary_insertion_targets(augment_relation_targets(augment_main_targets(train_arrays, vocab['purpose_target_vocab']), vocab['purpose_target_vocab']))
    valid_arrays = augment_secondary_insertion_targets(augment_relation_targets(augment_main_targets(valid_arrays, vocab['purpose_target_vocab']), vocab['purpose_target_vocab']))
    test_arrays = augment_secondary_insertion_targets(augment_relation_targets(augment_main_targets(test_arrays, vocab['purpose_target_vocab']), vocab['purpose_target_vocab']))

    zone_purpose_matrix, zone_support_binary, zone_coords, zone_idx_to_id = prepare_zone_support()
    num_zones = zone_purpose_matrix.shape[0]
    priors = {
        'purpose_step': compute_step_log_prior_matrix(train_arrays['y_purpose'], train_arrays['mask_purpose'], len(vocab['purpose_target_vocab'])),
        'origin_destination': compute_origin_destination_log_prior(train_arrays['step_cat'][:, :, 0], train_arrays['mask_dest_zone'], train_arrays['y_dest_zone'], train_arrays['mask_dest_zone'], num_zones),
        'depart_step': compute_step_log_prior_matrix(train_arrays['y_depart_bin24'], train_arrays['mask_depart_bin24'], num_time_classes + 1),
        'arrive_step': compute_step_log_prior_matrix(train_arrays['y_arrive_bin24'], train_arrays['mask_arrive_bin24'], num_time_classes + 1),
        'purpose_depart': compute_conditional_log_prior(train_arrays['y_purpose'], train_arrays['mask_purpose'], train_arrays['y_depart_bin24'], train_arrays['mask_depart_bin24'], len(vocab['purpose_target_vocab']), num_time_classes + 1),
        'purpose_arrive': compute_conditional_log_prior(train_arrays['y_purpose'], train_arrays['mask_purpose'], train_arrays['y_arrive_bin24'], train_arrays['mask_arrive_bin24'], len(vocab['purpose_target_vocab']), num_time_classes + 1),
    }
    zone_district_lookup = compute_zone_district_lookup(train_arrays, num_zones, len(vocab['origin_admin_district_vocab']))
    mode_priors = prepare_mode_priors(train_arrays, len(vocab['purpose_target_vocab']), len(vocab['mode_target_vocab']))
    priors['mode_step'] = mode_priors['step']
    priors['mode_purpose'] = mode_priors['purpose']
    priors['mode_distance'] = prepare_mode_distance_priors(train_arrays, zone_coords, len(vocab['mode_target_vocab']))
    priors['mode_transition'] = prepare_mode_transition_priors(train_arrays, len(vocab['mode_target_vocab']) + 1, len(vocab['mode_target_vocab']))
    priors['mode_usual_commute'] = prepare_mode_usual_commute_priors(train_arrays, len(vocab['usual_commute_mode_vocab']), len(vocab['mode_target_vocab']))

    model, zone_support_binary, zone_coords, zone_idx_to_id = build_model_and_support(config, vocab, priors, zone_support_binary, zone_coords, zone_idx_to_id, zone_purpose_matrix)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    dummy_batch = {key: tf.convert_to_tensor(value[:2]) for key, value in train_arrays.items()}
    _ = model(split_inputs(dummy_batch), training=False)

    class_weights = {
        'purpose': tf.constant(compute_class_weights(train_arrays['y_purpose'], train_arrays['mask_purpose'], len(vocab['purpose_target_vocab'])), dtype=tf.float32),
        'arrive': tf.constant(np.ones(num_time_classes + 1, dtype=np.float32), dtype=tf.float32),
        'depart': tf.constant(np.ones(num_time_classes + 1, dtype=np.float32), dtype=tf.float32),
        'mode': tf.constant(compute_class_weights(train_arrays['y_mode'], train_arrays['mask_mode'], len(vocab['mode_target_vocab']), power=0.65 if args.mode_schema == 'fine7' else 0.35), dtype=tf.float32),
    }
    default_mode_weight = args.mode_loss_weight if args.mode_schema == 'fine7' else min(args.mode_loss_weight, 1.4)
    base_loss_weights = {'purpose': 1.0, 'destination': args.destination_loss_weight, 'arrive': args.arrive_loss_weight, 'mode': default_mode_weight, 'depart': args.depart_loss_weight, 'duration': args.duration_loss_weight, 'dwell': args.dwell_loss_weight, 'depart_adjustment': args.depart_adjustment_loss_weight, 'mode_family': args.mode_family_loss_weight, 'arrive_coarse': args.arrive_coarse_loss_weight, 'depart_coarse': args.depart_coarse_loss_weight, 'main_anchor': args.main_anchor_loss_weight, 'main_destination_aux': args.main_destination_aux_loss_weight, 'relation_to_main': args.relation_to_main_loss_weight, 'secondary_insert': args.secondary_insert_loss_weight, 'resource_consistency': args.resource_consistency_loss_weight}

    train_ds = build_tf_dataset(train_arrays, args.batch_size, shuffle=True)
    valid_ds = build_tf_dataset(valid_arrays, args.batch_size, shuffle=False)
    ckpt = tf.train.Checkpoint(model=model)
    best_ckpt_prefix = str(run_dir / 'checkpoints' / 'best_ckpt')
    history_rows = []
    best_val = math.inf
    best_score = -math.inf
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        ss_prob = compute_schedule_sampling_prob(epoch, args.epochs, args.schedule_sampling_warmup, args.schedule_sampling_max)
        main_anchor_cond_factor = compute_ramp_factor(epoch, args.main_anchor_conditioning_warmup, args.main_anchor_conditioning_ramp) if args.use_main_anchor_conditioning else 0.0
        main_anchor_loss_factor = compute_ramp_factor(epoch, args.main_anchor_loss_warmup, args.main_anchor_loss_ramp) if args.use_main_anchor_conditioning else 0.0
        relation_loss_factor = compute_ramp_factor(epoch, args.relation_loss_warmup, args.relation_loss_ramp) if args.use_main_anchor_conditioning else 0.0
        secondary_insert_loss_factor = compute_ramp_factor(epoch, args.secondary_insert_loss_warmup, args.secondary_insert_loss_ramp) if args.use_main_anchor_conditioning else 0.0
        resource_cond_factor = compute_ramp_factor(epoch, args.resource_conditioning_warmup, args.resource_conditioning_ramp) if args.use_tour_resource_state else 0.0
        resource_loss_factor = compute_ramp_factor(epoch, args.resource_loss_warmup, args.resource_loss_ramp) if args.use_tour_resource_state else 0.0
        model.main_anchor_conditioning_factor.assign(main_anchor_cond_factor)
        model.resource_state_conditioning_factor.assign(resource_cond_factor)
        loss_weights = dict(base_loss_weights)
        loss_weights['main_anchor'] = base_loss_weights['main_anchor'] * main_anchor_loss_factor
        loss_weights['relation_to_main'] = base_loss_weights['relation_to_main'] * relation_loss_factor
        loss_weights['secondary_insert'] = base_loss_weights['secondary_insert'] * secondary_insert_loss_factor
        loss_weights['resource_consistency'] = base_loss_weights['resource_consistency'] * resource_loss_factor
        train_loss = run_epoch(model, optimizer, train_ds, class_weights, loss_weights, args.temporal_lambda, args.time_soft_sigma, args.time_distance_loss_weight, args.time_ordinal_loss_weight, ss_prob, zone_district_lookup, args.schedule_sampling_time_boost, args.schedule_sampling_late_boost)
        valid_loss = evaluate_loss(model, valid_ds, class_weights, loss_weights, args.temporal_lambda, args.time_soft_sigma, args.time_distance_loss_weight, args.time_ordinal_loss_weight)
        row = {'epoch': epoch, 'schedule_sampling_prob': ss_prob, 'main_anchor_cond_factor': main_anchor_cond_factor, 'main_anchor_loss_factor': main_anchor_loss_factor, 'relation_loss_factor': relation_loss_factor, 'secondary_insert_loss_factor': secondary_insert_loss_factor, 'resource_cond_factor': resource_cond_factor, 'resource_loss_factor': resource_loss_factor}
        row.update({f'train_{k}': v for k, v in train_loss.items()})
        row.update({f'valid_{k}': v for k, v in valid_loss.items()})
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / 'epoch_metrics.csv', index=False, encoding='utf-8-sig')
        checkpoint_metric_value = None
        if args.checkpoint_metric == 'loss':
            checkpoint_metric_value = -valid_loss['total']
        else:
            valid_use_autoregressive = (args.evaluation_mode == 'half_autoregressive')
            valid_inference = forward_batches(model, valid_arrays, args.batch_size, case_person_limit=0, autoregressive=valid_use_autoregressive, use_observed_origin=(valid_use_autoregressive and args.inference_origin == 'observed'), zone_district_lookup=zone_district_lookup)
            valid_metrics, _, _, _, _ = summarize_metrics('valid', valid_arrays, valid_inference, args.time_bin_size_hours, mode_eval_labels)
            checkpoint_metric_value = float(valid_metrics[args.checkpoint_metric])
            row[f'valid_metric_{args.checkpoint_metric}'] = checkpoint_metric_value
            history_rows[-1] = row
            pd.DataFrame(history_rows).to_csv(run_dir / 'epoch_metrics.csv', index=False, encoding='utf-8-sig')
        print(f"Epoch {epoch:02d} | ss_prob={ss_prob:.3f} | train_total={train_loss['total']:.4f} | valid_total={valid_loss['total']:.4f} | train_dest={train_loss['destination']:.4f} | valid_dest={valid_loss['destination']:.4f}")
        if checkpoint_metric_value > best_score:
            best_score = checkpoint_metric_value
            best_val = valid_loss['total']
            best_epoch = epoch
            patience_left = args.patience
            ckpt.write(best_ckpt_prefix)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'Early stopping at epoch {epoch}.')
                break

    ckpt.restore(best_ckpt_prefix).expect_partial()
    best_main_anchor_cond_factor = compute_ramp_factor(best_epoch, args.main_anchor_conditioning_warmup, args.main_anchor_conditioning_ramp) if args.use_main_anchor_conditioning else 0.0
    best_resource_cond_factor = compute_ramp_factor(best_epoch, args.resource_conditioning_warmup, args.resource_conditioning_ramp) if args.use_tour_resource_state else 0.0
    model.main_anchor_conditioning_factor.assign(best_main_anchor_cond_factor)
    model.resource_state_conditioning_factor.assign(best_resource_cond_factor)

    results = []
    figure_frames: dict[str, pd.DataFrame] = {}
    for split_name, arrays in [('train', train_arrays), ('valid', valid_arrays), ('test', test_arrays)]:
        use_autoregressive = (split_name in {'valid', 'test'} and args.evaluation_mode == 'half_autoregressive')
        inference = forward_batches(model, arrays, args.batch_size, case_person_limit=40 if split_name == 'test' else 10, autoregressive=use_autoregressive, use_observed_origin=(use_autoregressive and args.inference_origin == 'observed'), zone_district_lookup=zone_district_lookup)
        metrics, by_step_df, topk_df, chain_df, behavior_df = summarize_metrics(split_name, arrays, inference, args.time_bin_size_hours, mode_eval_labels)
        main_df = compute_main_activity_metrics(arrays, inference, vocab['purpose_target_vocab'], split_name)
        results.append(metrics)
        figure_frames[f'{split_name}_metrics_by_step'] = by_step_df
        figure_frames[f'{split_name}_destination_topk'] = topk_df
        figure_frames[f'{split_name}_chain_level_metrics'] = chain_df
        figure_frames[f'{split_name}_behavioral_consistency'] = behavior_df
        figure_frames[f'{split_name}_main_activity_metrics'] = main_df
        if split_name == 'test':
            save_case_outputs(run_dir, inference, zone_idx_to_id)

    results_df = pd.DataFrame(results)
    results_df.to_csv(run_dir / 'test_metrics_overall.csv', index=False, encoding='utf-8-sig')
    for name, frame in figure_frames.items():
        frame.to_csv(run_dir / 'figure_data' / f'{name}.csv', index=False, encoding='utf-8-sig')
    write_png_figures(run_dir, pd.DataFrame(history_rows), results_df, figure_frames)

    test_metrics = results_df[results_df['split'] == 'test'].iloc[0].to_dict()
    append_experiment_registry(run_dir, config, best_epoch, test_metrics)
    print(results_df.to_string(index=False))
    print(f'Best epoch: {best_epoch}')
    print(f'Outputs saved to: {run_dir}')


if __name__ == '__main__':
    main()












