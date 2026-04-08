from __future__ import annotations

from typing import Any

import tensorflow as tf


class TourInterpretableGraphACTNN(tf.keras.Model):
    def __init__(
        self,
        num_zones: int,
        num_purpose: int,
        num_time: int,
        num_mode: int,
        num_gender: int,
        num_occupation: int,
        num_schooling: int,
        num_housing: int,
        num_usual_commute: int,
        num_district: int,
        zone_feature_matrix,
        zone_purpose_matrix,
        zone_coord_matrix,
        purpose_step_prior_matrix,
        origin_destination_prior_matrix,
        depart_step_prior_matrix,
        arrive_step_prior_matrix,
        purpose_depart_prior_matrix,
        purpose_arrive_prior_matrix,
        mode_step_prior_matrix,
        purpose_mode_prior_matrix,
        mode_distance_prior_matrix,
        mode_transition_prior_matrix,
        mode_usual_commute_prior_matrix,
        origin_candidate_log_mask,
        config: dict[str, Any],
    ) -> None:
        super().__init__(name='tour_interpretable_graph_actnn')
        self.num_zones = int(num_zones)
        self.num_purpose = int(num_purpose)
        self.num_time = int(num_time)
        self.num_mode = int(num_mode)
        self.num_relation = int(config.get('num_relation', 6))
        self.home_purpose_id = int(config.get('home_purpose_id', 1))
        self.work_purpose_id = int(config.get('work_purpose_id', 2))
        self.bos_purpose_id = int(config.get('bos_purpose_id', self.num_purpose))
        self.bos_mode_id = int(config.get('bos_mode_id', self.num_mode))
        self.hidden_dim = int(config.get('hidden_dim', 128))
        self.dropout_rate = float(config.get('dropout', 0.15))
        self.max_steps = int(config.get('max_steps', 16))
        self.use_step_embedding = bool(config.get('use_step_embedding', True))
        self.destination_context_k = int(config.get('destination_context_k', 8))
        self.destination_context_topk_alpha = float(config.get('destination_context_topk_alpha', 0.7))
        self.use_duration_depart = bool(config.get('use_duration_depart', False))
        self.duration_depart_blend = float(config.get('duration_depart_blend', 0.6))
        self.use_depart_adjustment = bool(config.get('use_depart_adjustment', True))
        self.depart_dwell_bias_scale = float(config.get('depart_dwell_bias_scale', 0.25))
        self.predict_depart_first = bool(config.get('predict_depart_first', False))
        self.use_main_anchor_conditioning = bool(config.get('use_main_anchor_conditioning', False))
        self.use_tour_resource_state = bool(config.get('use_tour_resource_state', False))
        self.use_relation_to_main_conditioning = bool(config.get('use_relation_to_main_conditioning', True))
        self.use_secondary_local_destination = bool(config.get('use_secondary_local_destination', True))
        self.use_interpretable_modules = bool(config.get('use_interpretable_modules', True))
        self.main_anchor_conditioning_factor = self.add_weight(
            name='main_anchor_conditioning_factor',
            shape=(),
            initializer=tf.keras.initializers.Constant(float(config.get('main_anchor_conditioning_factor', 1.0))),
            trainable=False,
        )
        self.resource_state_conditioning_factor = self.add_weight(
            name='resource_state_conditioning_factor',
            shape=(),
            initializer=tf.keras.initializers.Constant(float(config.get('resource_state_conditioning_factor', 1.0))),
            trainable=False,
        )
        self.walk_mode_id = int(config.get('walk_mode_id', 0))
        self.bike_mode_id = int(config.get('bike_mode_id', 0))
        self.bus_mode_id = int(config.get('bus_mode_id', 0))
        self.metro_mode_id = int(config.get('metro_mode_id', 0))
        self.taxi_mode_id = int(config.get('taxi_mode_id', 0))
        self.car_mode_id = int(config.get('car_mode_id', 0))
        zone_embed_dim = int(config.get('zone_embed_dim', 32))
        zone_fixed_dim = int(config.get('zone_fixed_dim', 24))
        general_embed_dim = int(config.get('general_embed_dim', 8))

        self.zone_feature_matrix = tf.constant(zone_feature_matrix, dtype=tf.float32)
        self.zone_purpose_matrix = tf.constant(zone_purpose_matrix, dtype=tf.float32)
        self.zone_support_binary = tf.cast(self.zone_purpose_matrix > 0, tf.float32)
        self.zone_coord_matrix = tf.constant(zone_coord_matrix, dtype=tf.float32)
        self.purpose_step_prior_matrix = tf.constant(purpose_step_prior_matrix, dtype=tf.float32)
        self.origin_destination_prior_matrix = tf.constant(origin_destination_prior_matrix, dtype=tf.float32)
        self.depart_step_prior_matrix = tf.constant(depart_step_prior_matrix, dtype=tf.float32)
        self.arrive_step_prior_matrix = tf.constant(arrive_step_prior_matrix, dtype=tf.float32)
        self.purpose_depart_prior_matrix = tf.constant(purpose_depart_prior_matrix, dtype=tf.float32)
        self.purpose_arrive_prior_matrix = tf.constant(purpose_arrive_prior_matrix, dtype=tf.float32)
        self.mode_step_prior_matrix = tf.constant(mode_step_prior_matrix, dtype=tf.float32)
        self.purpose_mode_prior_matrix = tf.constant(purpose_mode_prior_matrix, dtype=tf.float32)
        self.mode_distance_prior_matrix = tf.constant(mode_distance_prior_matrix, dtype=tf.float32)
        self.mode_transition_prior_matrix = tf.constant(mode_transition_prior_matrix, dtype=tf.float32)
        self.mode_usual_commute_prior_matrix = tf.constant(mode_usual_commute_prior_matrix, dtype=tf.float32)
        self.origin_candidate_log_mask = tf.constant(origin_candidate_log_mask, dtype=tf.float32)
        mode_family_map = [[0.0] * self.num_mode for _ in range(4)]
        if self.num_mode > 2:
            mode_family_map[0][1] = 1.0
            mode_family_map[0][2] = 1.0
        if self.num_mode > 4:
            mode_family_map[1][3] = 1.0
            mode_family_map[1][4] = 1.0
        if self.num_mode > 6:
            mode_family_map[2][5] = 1.0
            mode_family_map[2][6] = 1.0
        if self.num_mode > 7:
            mode_family_map[3][7] = 1.0
        self.mode_family_to_mode_matrix = tf.constant(mode_family_map, dtype=tf.float32)
        mode_family_log_mask = [[-1e9] * self.num_mode for _ in range(4)]
        for fam_idx, fam_row in enumerate(mode_family_map):
            for mode_idx, flag in enumerate(fam_row):
                if flag > 0:
                    mode_family_log_mask[fam_idx][mode_idx] = 0.0
        self.mode_family_log_mask = tf.constant(mode_family_log_mask, dtype=tf.float32)

        coarse_time_map = [[0.0] * self.num_time for _ in range(4)]
        num_valid_time = max(self.num_time - 1, 1)
        for fine_label in range(1, self.num_time):
            fine_zero = fine_label - 1
            coarse_idx = min((fine_zero * 4) // num_valid_time, 3)
            coarse_time_map[coarse_idx][fine_label] = 1.0
        self.coarse_time_to_fine_matrix = tf.constant(coarse_time_map, dtype=tf.float32)

        self.zone_embedding = tf.keras.layers.Embedding(self.num_zones, zone_embed_dim, mask_zero=True, name='zone_embedding')
        self.zone_feature_proj = tf.keras.layers.Dense(zone_fixed_dim, activation='relu', name='zone_feature_proj')
        self.gender_embedding = tf.keras.layers.Embedding(num_gender, general_embed_dim, name='gender_embedding')
        self.occupation_embedding = tf.keras.layers.Embedding(num_occupation, general_embed_dim * 2, name='occupation_embedding')
        self.schooling_embedding = tf.keras.layers.Embedding(num_schooling, general_embed_dim, name='schooling_embedding')
        self.housing_embedding = tf.keras.layers.Embedding(num_housing, general_embed_dim, name='housing_embedding')
        self.usual_commute_embedding = tf.keras.layers.Embedding(num_usual_commute, general_embed_dim, name='usual_commute_embedding')
        self.prev_purpose_embedding = tf.keras.layers.Embedding(num_purpose + 1, general_embed_dim, name='prev_purpose_embedding')
        self.prev_mode_embedding = tf.keras.layers.Embedding(num_mode + 1, general_embed_dim, name='prev_mode_embedding')
        self.district_embedding = tf.keras.layers.Embedding(num_district, general_embed_dim, name='district_embedding')
        self.step_index_embedding = tf.keras.layers.Embedding(self.max_steps, general_embed_dim, name='step_index_embedding')
        self.static_num_proj = tf.keras.layers.Dense(16, activation='relu', name='static_num_proj')
        self.step_num_proj = tf.keras.layers.Dense(16, activation='relu', name='step_num_proj')
        self.static_context = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
        ], name='static_context')
        self.sequence_input_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='sequence_input_proj')
        self.sequence_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_dim // 2, return_sequences=True, dropout=self.dropout_rate),
            merge_mode='concat',
            name='bilstm_encoder',
        )
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=int(config.get('num_heads', 4)),
            key_dim=self.hidden_dim // int(config.get('num_heads', 4)),
            dropout=self.dropout_rate,
            name='sequence_attention',
        )
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='attention_norm')
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim * 2, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.hidden_dim),
        ], name='ffn')
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ffn_norm')

        self.main_step_head = tf.keras.layers.Dense(1, name='main_step_head')
        self.main_context_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='main_context_proj')
        self.main_global_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='main_global_proj')
        self.main_purpose_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_purpose),
        ], name='main_purpose_head')
        self.main_destination_query = tf.keras.layers.Dense(zone_embed_dim + zone_fixed_dim, name='main_destination_query')
        self.main_destination_query_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='main_destination_query_norm')
        self.main_destination_refine_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='main_destination_refine_proj')
        self.main_destination_refine_query = tf.keras.layers.Dense(zone_embed_dim + zone_fixed_dim, name='main_destination_refine_query')
        self.main_destination_refine_query_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='main_destination_refine_query_norm')
        self.main_destination_local_query = tf.keras.layers.Dense(zone_embed_dim + zone_fixed_dim, name='main_destination_local_query')
        self.main_destination_local_query_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='main_destination_local_query_norm')
        self.main_destination_mix_gate = tf.keras.layers.Dense(1, activation='sigmoid', name='main_destination_mix_gate')
        self.main_arrive_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_time),
        ], name='main_arrive_head')
        self.main_mode_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_mode),
        ], name='main_mode_head')
        self.main_depart_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_time),
        ], name='main_depart_head')
        self.main_bias_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='main_bias_proj')
        self.relation_to_main_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_relation),
        ], name='relation_to_main_head')
        self.secondary_insert_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(4),
        ], name='secondary_insert_head')
        self.secondary_window_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(4),
        ], name='secondary_window_head')
        self.secondary_insert_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='secondary_insert_proj')
        self.secondary_local_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='secondary_local_proj')
        self.secondary_destination_bias_head = tf.keras.layers.Dense(self.num_zones, name='secondary_destination_bias_head')
        self.secondary_destination_gate = tf.keras.layers.Dense(1, activation='sigmoid', name='secondary_destination_gate')
        self.secondary_destination_query = tf.keras.layers.Dense(zone_embed_dim + zone_fixed_dim, name='secondary_destination_query')
        self.secondary_destination_query_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='secondary_destination_query_norm')
        self.secondary_destination_mix_gate = tf.keras.layers.Dense(1, activation='sigmoid', name='secondary_destination_mix_gate')
        self.secondary_block_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='secondary_block_proj')
        self.secondary_block_purpose_head = tf.keras.layers.Dense(self.num_purpose, name='secondary_block_purpose_head')
        self.secondary_block_arrive_head = tf.keras.layers.Dense(self.num_time, name='secondary_block_arrive_head')
        self.secondary_purpose_bias_head = tf.keras.layers.Dense(self.num_purpose, name='secondary_purpose_bias_head')
        self.secondary_arrive_bias_head = tf.keras.layers.Dense(self.num_time, name='secondary_arrive_bias_head')
        self.secondary_phase_token_matrix = self.add_weight(
            name='secondary_phase_token_matrix',
            shape=(4, self.hidden_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.relation_token_matrix = self.add_weight(
            name='relation_token_matrix',
            shape=(self.num_relation, self.hidden_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.purpose_utility_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='purpose_utility_proj')
        self.destination_accessibility_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='destination_accessibility_proj')
        self.scheduling_pressure_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='scheduling_pressure_proj')
        self.dwell_context_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='dwell_context_proj')
        self.mode_feasibility_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='mode_feasibility_proj')
        self.departure_adjustment_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='departure_adjustment_proj')
        self.purpose_plain_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='purpose_plain_proj')
        self.destination_plain_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='destination_plain_proj')
        self.arrive_plain_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='arrive_plain_proj')
        self.mode_plain_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='mode_plain_proj')
        self.dwell_plain_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='dwell_plain_proj')
        self.depart_plain_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='depart_plain_proj')
        self.resource_state_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='resource_state_proj')
        self.resource_mode_bias_head = tf.keras.layers.Dense(self.num_mode, name='resource_mode_bias_head')

        self.purpose_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_purpose),
        ], name='purpose_head')
        self.destination_query = tf.keras.layers.Dense(zone_embed_dim + zone_fixed_dim, name='destination_query')
        self.destination_query_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='destination_query_norm')
        self.arrive_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_time),
        ], name='arrive_head')
        self.mode_family_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(4),
        ], name='mode_family_head')
        self.mode_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_mode),
        ], name='mode_head')
        self.arrive_coarse_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(4),
        ], name='arrive_coarse_head')
        self.depart_coarse_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(4),
        ], name='depart_coarse_head')
        self.depart_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_time),
        ], name='depart_head')
        self.duration_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_time),
        ], name='duration_head')
        self.dwell_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_time),
        ], name='dwell_head')
        self.depart_dwell_bias_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_time),
        ], name='depart_dwell_bias_head')
        self.depart_blend_gate_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name='depart_blend_gate_head')
        self.mode_context_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='mode_context_proj')
        self.mode_distance_proj = tf.keras.layers.Dense(16, activation='relu', name='mode_distance_proj')
        self.depart_arrive_summary_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='depart_arrive_summary_proj')
        self.depart_arrive_gate = tf.keras.layers.Dense(self.hidden_dim, activation='sigmoid', name='depart_arrive_gate')
        self.depart_context_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='depart_context_proj')

        self.purpose_step_prior_scale = self.add_weight(name='purpose_step_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.9), trainable=True)
        self.arrive_step_prior_scale = self.add_weight(name='arrive_step_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.8), trainable=True)
        self.depart_step_prior_scale = self.add_weight(name='depart_step_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.8), trainable=True)
        self.purpose_arrive_prior_scale = self.add_weight(name='purpose_arrive_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.9), trainable=True)
        self.purpose_depart_prior_scale = self.add_weight(name='purpose_depart_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.9), trainable=True)
        self.purpose_destination_scale = self.add_weight(name='purpose_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(0.7), trainable=True)
        self.origin_destination_scale = self.add_weight(name='origin_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(1.4), trainable=True)
        self.destination_support_scale = self.add_weight(name='destination_support_scale', shape=(), initializer=tf.keras.initializers.Constant(1.2), trainable=True)
        self.home_destination_scale = self.add_weight(name='home_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True)
        self.work_destination_scale = self.add_weight(name='work_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(1.8), trainable=True)
        self.first_step_work_purpose_scale = self.add_weight(name='first_step_work_purpose_scale', shape=(), initializer=tf.keras.initializers.Constant(1.3), trainable=True)
        self.first_step_work_destination_scale = self.add_weight(name='first_step_work_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(1.5), trainable=True)
        self.origin_candidate_scale = self.add_weight(name='origin_candidate_scale', shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.mode_step_prior_scale = self.add_weight(name='mode_step_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.4), trainable=True)
        self.mode_purpose_prior_scale = self.add_weight(name='mode_purpose_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)
        self.mode_distance_prior_scale = self.add_weight(name='mode_distance_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(0.8), trainable=True)
        self.mode_transition_prior_scale = self.add_weight(name='mode_transition_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(1.1), trainable=True)
        self.mode_usual_commute_prior_scale = self.add_weight(name='mode_usual_commute_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.mode_family_prior_scale = self.add_weight(name='mode_family_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.main_destination_bias_scale = self.add_weight(name='main_destination_bias_scale', shape=(), initializer=tf.keras.initializers.Constant(0.4), trainable=True)
        self.main_destination_origin_scale = self.add_weight(name='main_destination_origin_scale', shape=(), initializer=tf.keras.initializers.Constant(0.8), trainable=True)
        self.main_destination_candidate_scale = self.add_weight(name='main_destination_candidate_scale', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)
        self.main_destination_refine_scale = self.add_weight(name='main_destination_refine_scale', shape=(), initializer=tf.keras.initializers.Constant(0.8), trainable=True)
        self.main_destination_local_scale = self.add_weight(name='main_destination_local_scale', shape=(), initializer=tf.keras.initializers.Constant(0.55), trainable=True)
        self.secondary_destination_bias_scale = self.add_weight(name='secondary_destination_bias_scale', shape=(), initializer=tf.keras.initializers.Constant(0.35), trainable=True)
        self.secondary_destination_local_scale = self.add_weight(name='secondary_destination_local_scale', shape=(), initializer=tf.keras.initializers.Constant(0.40), trainable=True)
        self.secondary_phase_destination_scale = self.add_weight(name='secondary_phase_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(0.9), trainable=True)
        self.resource_mode_bias_scale = self.add_weight(name='resource_mode_bias_scale', shape=(), initializer=tf.keras.initializers.Constant(0.7), trainable=True)
        depart_transition = [[[0.0 for _ in range(self.num_time)] for _ in range(self.num_time)] for _ in range(self.num_time)]
        for arrive_label in range(self.num_time):
            for duration_label in range(self.num_time):
                if arrive_label == 0 or duration_label == 0:
                    depart_transition[arrive_label][duration_label][0] = 1.0
                else:
                    duration_offset = duration_label - 1
                    depart_label = max(arrive_label - duration_offset, 1)
                    depart_transition[arrive_label][duration_label][depart_label] = 1.0
        self.depart_from_arrive_duration = tf.constant(depart_transition, dtype=tf.float32)

    def _scan_resource_state(self, origin_zone_seq: tf.Tensor, prev_mode_seq: tf.Tensor, home_zone_idx: tf.Tensor, resource_mode_id: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(origin_zone_seq)[0]
        time_steps = self.max_steps
        if resource_mode_id <= 0:
            zeros = tf.zeros((batch_size, time_steps), dtype=tf.float32)
            return zeros, zeros, zeros
        parked = tf.where(home_zone_idx > 0, home_zone_idx, tf.zeros_like(home_zone_idx))
        available = tf.cast(parked > 0, tf.float32)
        available_list = []
        retrieval_list = []
        dist_norm_list = []
        for t in range(self.max_steps):
            curr_origin = origin_zone_seq[:, t]
            if t > 0:
                prev_mode_t = prev_mode_seq[:, t]
                prev_origin_t = origin_zone_seq[:, t - 1]
                used_resource = tf.cast(prev_mode_t == resource_mode_id, tf.int32)
                parked_keep = tf.where(available > 0.5, prev_origin_t, parked)
                parked = tf.where(used_resource > 0, curr_origin, parked_keep)
            available = tf.cast((parked > 0) & (parked == curr_origin), tf.float32)
            retrieval = tf.cast((parked > 0) & (parked != curr_origin), tf.float32)
            parked_coords = tf.gather(self.zone_coord_matrix, parked)
            origin_coords_t = tf.gather(self.zone_coord_matrix, curr_origin)
            delta_lon = origin_coords_t[:, 0] - parked_coords[:, 0]
            delta_lat = origin_coords_t[:, 1] - parked_coords[:, 1]
            dist_km = tf.sqrt(tf.maximum(tf.square(delta_lon * 111.32) + tf.square(delta_lat * 110.57), 1e-6))
            dist_norm = tf.clip_by_value(dist_km / 20.0, 0.0, 1.0)
            available_list.append(available)
            retrieval_list.append(retrieval)
            dist_norm_list.append(dist_norm)
        return tf.stack(available_list, axis=1), tf.stack(retrieval_list, axis=1), tf.stack(dist_norm_list, axis=1)
    def _mask_pad_class(self, logits: tf.Tensor) -> tf.Tensor:
        neg = tf.cast(-1e9, logits.dtype)
        masked_first = tf.fill(tf.shape(logits[..., :1]), neg)
        return tf.concat([masked_first, logits[..., 1:]], axis=-1)

    def _zone_context(self, zone_idx: tf.Tensor) -> tf.Tensor:
        trainable_embed = self.zone_embedding(zone_idx)
        fixed_features = tf.gather(self.zone_feature_matrix, zone_idx)
        fixed_embed = self.zone_feature_proj(fixed_features)
        return tf.concat([trainable_embed, fixed_embed], axis=-1)

    def _attention_mask(self, seq_mask: tf.Tensor) -> tf.Tensor:
        mask_bool = tf.cast(seq_mask > 0, tf.bool)
        return tf.logical_and(mask_bool[:, :, None], mask_bool[:, None, :])

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False, return_attention: bool = False) -> dict[str, tf.Tensor]:
        static_cat = tf.cast(inputs['static_cat'], tf.int32)
        static_num = tf.cast(inputs['static_num'], tf.float32)
        step_cat = tf.cast(inputs['step_cat'], tf.int32)
        step_num = tf.cast(inputs['step_num'], tf.float32)
        seq_mask = tf.cast(inputs['seq_mask'], tf.float32)

        home_zone_idx = static_cat[:, 5]
        work_zone_idx = static_cat[:, 6]
        static_parts = [
            self.gender_embedding(static_cat[:, 0]),
            self.occupation_embedding(static_cat[:, 1]),
            self.schooling_embedding(static_cat[:, 2]),
            self.housing_embedding(static_cat[:, 3]),
            self.usual_commute_embedding(static_cat[:, 4]),
            self._zone_context(home_zone_idx),
            self._zone_context(work_zone_idx),
            self.static_num_proj(static_num),
        ]
        static_context = self.static_context(tf.concat(static_parts, axis=-1), training=training)
        time_steps = tf.shape(step_cat)[1]
        batch_size = tf.shape(step_cat)[0]
        static_context_tiled = tf.repeat(static_context[:, tf.newaxis, :], repeats=time_steps, axis=1)

        origin_zone_context = self._zone_context(step_cat[:, :, 0])
        prev_purpose_context = self.prev_purpose_embedding(step_cat[:, :, 1])
        prev_mode_context = self.prev_mode_embedding(step_cat[:, :, 2])
        district_context = self.district_embedding(step_cat[:, :, 3])
        if self.use_step_embedding:
            step_ids = tf.tile(tf.range(time_steps, dtype=tf.int32)[tf.newaxis, :], [tf.shape(step_cat)[0], 1])
            step_index_context = self.step_index_embedding(step_ids)
        else:
            step_index_context = tf.zeros_like(district_context)
        step_num_context = self.step_num_proj(step_num)

        x = tf.concat([
            static_context_tiled,
            origin_zone_context,
            prev_purpose_context,
            prev_mode_context,
            district_context,
            step_num_context,
            step_index_context,
        ], axis=-1)
        x = self.sequence_input_proj(x)
        x = self.sequence_dropout(x, training=training)
        x = self.encoder(x, mask=tf.cast(seq_mask > 0, tf.bool), training=training)

        if return_attention:
            attn_out, attn_scores = self.self_attention(x, x, attention_mask=self._attention_mask(seq_mask), return_attention_scores=True, training=training)
        else:
            attn_out = self.self_attention(x, x, attention_mask=self._attention_mask(seq_mask), training=training)
            attn_scores = None
        x = self.attention_norm(x + attn_out)
        x = self.ffn_norm(x + self.ffn(x, training=training))
        x = x * seq_mask[:, :, tf.newaxis]
        global_context = tf.reduce_sum(x, axis=1) / (tf.reduce_sum(seq_mask, axis=1, keepdims=True) + 1e-6)

        if self.use_main_anchor_conditioning:
            main_step_logits = tf.squeeze(self.main_step_head(x), axis=-1)
            main_step_logits += (1.0 - seq_mask) * tf.cast(-1e9, x.dtype)
            main_step_probs = tf.nn.softmax(main_step_logits, axis=-1)
            main_context = tf.reduce_sum(x * main_step_probs[:, :, tf.newaxis], axis=1)
            main_context = self.main_context_proj(tf.concat([main_context, self.main_global_proj(tf.concat([global_context, static_context], axis=-1))], axis=-1))
            main_purpose_logits = self._mask_pad_class(self.main_purpose_head(main_context, training=training))
            main_purpose_probs = tf.nn.softmax(main_purpose_logits, axis=-1)
            zone_repr = self._zone_context(tf.range(self.num_zones, dtype=tf.int32))
            main_destination_query = self.main_destination_query_norm(
                self.main_destination_query(tf.concat([main_context, main_purpose_probs], axis=-1))
            )
            main_destination_logits = tf.einsum('bd,zd->bz', main_destination_query, zone_repr)
            main_origin_probs = tf.reduce_sum(tf.one_hot(step_cat[:, :, 0], depth=self.num_zones, dtype=x.dtype) * main_step_probs[:, :, tf.newaxis], axis=1)
            main_origin_context = tf.einsum('bo,od->bd', main_origin_probs, zone_repr)
            home_zone_context = self._zone_context(home_zone_idx)
            work_zone_context = self._zone_context(work_zone_idx)
            main_destination_refine_context = self.main_destination_refine_proj(tf.concat([
                main_context,
                main_purpose_probs,
                main_origin_context,
                home_zone_context,
                work_zone_context,
                global_context,
                static_context,
            ], axis=-1))
            main_destination_refine_query = self.main_destination_refine_query_norm(
                self.main_destination_refine_query(main_destination_refine_context)
            )
            main_destination_local_query = self.main_destination_local_query_norm(
                self.main_destination_local_query(tf.concat([
                    main_destination_refine_context,
                    main_origin_context,
                    home_zone_context,
                    work_zone_context,
                ], axis=-1))
            )
            main_destination_mix_gate = self.main_destination_mix_gate(tf.concat([
                main_context,
                main_origin_context,
                main_purpose_probs,
            ], axis=-1))
            main_destination_logits += self.main_destination_refine_scale * tf.einsum('bd,zd->bz', main_destination_refine_query, zone_repr)
            main_destination_logits += self.main_destination_local_scale * main_destination_mix_gate * tf.einsum('bd,zd->bz', main_destination_local_query, zone_repr)
            main_destination_logits += self.main_destination_origin_scale * tf.einsum('bo,oz->bz', main_origin_probs, self.origin_destination_prior_matrix)
            main_destination_logits += self.main_destination_candidate_scale * tf.einsum('bo,oz->bz', main_origin_probs, self.origin_candidate_log_mask)
            main_destination_logits += self.purpose_destination_scale * tf.einsum('bp,zp->bz', main_purpose_probs, self.zone_purpose_matrix)
            main_destination_logits += self.destination_support_scale * tf.einsum('bp,zp->bz', main_purpose_probs, self.zone_support_binary)
            home_one_hot_main = tf.one_hot(home_zone_idx, depth=self.num_zones, dtype=tf.float32)
            work_one_hot_main = tf.one_hot(work_zone_idx, depth=self.num_zones, dtype=tf.float32)
            main_destination_logits += self.home_destination_scale * main_purpose_probs[:, self.home_purpose_id:self.home_purpose_id + 1] * home_one_hot_main
            main_destination_logits += self.work_destination_scale * main_purpose_probs[:, self.work_purpose_id:self.work_purpose_id + 1] * work_one_hot_main
            main_destination_logits = self._mask_pad_class(main_destination_logits)
            main_destination_probs = tf.nn.softmax(main_destination_logits, axis=-1)
            main_dest_topk = tf.math.top_k(main_destination_logits, k=min(4, self.num_zones))
            main_dest_topk_weights = tf.nn.softmax(1.5 * main_dest_topk.values, axis=-1)
            main_destination_topk_context = tf.reduce_sum(main_dest_topk_weights[..., tf.newaxis] * tf.gather(zone_repr, main_dest_topk.indices), axis=-2)
            main_destination_context = 0.65 * tf.einsum('bz,zd->bd', main_destination_probs, zone_repr) + 0.35 * main_destination_topk_context
            main_destination_purpose = tf.einsum('bz,zp->bp', main_destination_probs, self.zone_purpose_matrix)
            main_arrive_logits = self._mask_pad_class(self.main_arrive_head(tf.concat([main_context, main_purpose_probs, main_destination_context, main_destination_purpose], axis=-1), training=training))
            main_arrive_probs = tf.nn.softmax(main_arrive_logits, axis=-1)
            main_mode_logits = self._mask_pad_class(self.main_mode_head(tf.concat([main_context, main_purpose_probs, main_destination_context, main_arrive_probs], axis=-1), training=training))
            main_mode_probs = tf.nn.softmax(main_mode_logits, axis=-1)
            main_depart_logits = self._mask_pad_class(self.main_depart_head(tf.concat([main_context, main_purpose_probs, main_destination_context, main_arrive_probs, main_mode_probs], axis=-1), training=training))
            main_depart_probs = tf.nn.softmax(main_depart_logits, axis=-1)

            expected_main_step = tf.reduce_sum(main_step_probs * tf.cast(step_ids, tf.float32), axis=-1, keepdims=True) / tf.maximum(tf.cast(self.max_steps - 1, tf.float32), 1.0)
            relative_to_main = (tf.cast(step_ids, tf.float32) / tf.maximum(tf.cast(self.max_steps - 1, tf.float32), 1.0)) - expected_main_step
            main_time_values = tf.maximum(tf.cast(tf.range(self.num_time), tf.float32) - 1.0, 0.0)
            main_arrive_expect = tf.reduce_sum(main_arrive_probs * main_time_values[tf.newaxis, :], axis=-1, keepdims=True) / tf.maximum(tf.cast(self.num_time - 2, tf.float32), 1.0)
            main_depart_expect = tf.reduce_sum(main_depart_probs * main_time_values[tf.newaxis, :], axis=-1, keepdims=True) / tf.maximum(tf.cast(self.num_time - 2, tf.float32), 1.0)
            repeated_main_context = tf.repeat(main_context[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_purpose = tf.repeat(main_purpose_probs[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_dest_context = tf.repeat(main_destination_context[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_dest_purpose = tf.repeat(main_destination_purpose[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_mode = tf.repeat(main_mode_probs[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_arrive = tf.repeat(main_arrive_probs[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_depart = tf.repeat(main_depart_probs[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_arrive_expect = tf.repeat(main_arrive_expect[:, tf.newaxis, :], repeats=time_steps, axis=1)
            repeated_main_depart_expect = tf.repeat(main_depart_expect[:, tf.newaxis, :], repeats=time_steps, axis=1)
            main_relation_features = tf.concat([relative_to_main[:, :, tf.newaxis], repeated_main_arrive_expect, repeated_main_depart_expect], axis=-1)
            main_anchor_bias = self.main_bias_proj(tf.concat([
                repeated_main_context,
                repeated_main_purpose,
                repeated_main_dest_context,
                repeated_main_dest_purpose,
                repeated_main_mode,
                repeated_main_arrive,
                repeated_main_depart,
                main_relation_features,
            ], axis=-1))
        else:
            zone_repr = self._zone_context(tf.range(self.num_zones, dtype=tf.int32))
            batch_size = tf.shape(x)[0]
            main_step_logits = tf.zeros((batch_size, time_steps), dtype=x.dtype)
            main_purpose_logits = tf.zeros((batch_size, self.num_purpose), dtype=x.dtype)
            main_destination_logits = tf.zeros((batch_size, self.num_zones), dtype=x.dtype)
            main_arrive_logits = tf.zeros((batch_size, self.num_time), dtype=x.dtype)
            main_mode_logits = tf.zeros((batch_size, self.num_mode), dtype=x.dtype)
            main_depart_logits = tf.zeros((batch_size, self.num_time), dtype=x.dtype)
            main_destination_purpose = tf.zeros((batch_size, tf.shape(self.zone_purpose_matrix)[-1]), dtype=x.dtype)
            main_destination_probs = tf.one_hot(tf.zeros((batch_size,), dtype=tf.int32), depth=self.num_zones, dtype=x.dtype)
            repeated_main_context = tf.zeros_like(x)
            repeated_main_purpose = tf.zeros((batch_size, time_steps, self.num_purpose), dtype=x.dtype)
            repeated_main_dest_context = tf.zeros_like(origin_zone_context)
            main_dest_topk = tf.math.top_k(main_destination_logits, k=1)
            main_dest_topk_weights = tf.ones_like(main_dest_topk.values, dtype=x.dtype)
            repeated_main_mode = tf.zeros((batch_size, time_steps, self.num_mode), dtype=x.dtype)
            repeated_main_dest_purpose = tf.zeros((batch_size, time_steps, tf.shape(self.zone_purpose_matrix)[-1]), dtype=x.dtype)
            repeated_main_arrive_expect = tf.zeros((batch_size, time_steps, 1), dtype=x.dtype)
            repeated_main_depart_expect = tf.zeros((batch_size, time_steps, 1), dtype=x.dtype)
            relative_to_main = tf.zeros((batch_size, time_steps), dtype=x.dtype)
            main_anchor_bias = tf.zeros_like(x)

        relation_input = tf.concat([x, main_anchor_bias, step_index_context], axis=-1)
        relation_to_main_logits = self._mask_pad_class(self.relation_to_main_head(relation_input, training=training))
        relation_probs = tf.nn.softmax(relation_to_main_logits, axis=-1)
        relation_context = tf.einsum('btr,rd->btd', relation_probs, self.relation_token_matrix)
        if not self.use_relation_to_main_conditioning:
            relation_context = tf.zeros_like(relation_context)
            relation_probs = tf.concat([
                tf.ones_like(relation_probs[:, :, :1]),
                tf.zeros_like(relation_probs[:, :, 1:]),
            ], axis=-1)
        secondary_window_features = tf.concat([
            relative_to_main[:, :, tf.newaxis],
            repeated_main_arrive_expect,
            repeated_main_depart_expect,
            step_num[:, :, 1:3],
            step_num[:, :, 3:4],
        ], axis=-1)
        secondary_window_logits = self.secondary_window_head(secondary_window_features, training=training)
        secondary_insert_logits = self.secondary_insert_head(tf.concat([x, main_anchor_bias, relation_context, step_index_context], axis=-1), training=training) + secondary_window_logits
        secondary_insert_probs = tf.nn.softmax(secondary_insert_logits, axis=-1)
        secondary_phase_prior = tf.stack([
            tf.clip_by_value(1.0 - (relation_probs[:, :, 1] + relation_probs[:, :, 3] + relation_probs[:, :, 4]), 0.0, 1.0),
            relation_probs[:, :, 1],
            relation_probs[:, :, 3],
            relation_probs[:, :, 4],
        ], axis=-1)
        secondary_window_probs = tf.nn.softmax(secondary_window_logits, axis=-1)
        secondary_phase_probs = 0.45 * secondary_insert_probs + 0.30 * secondary_phase_prior + 0.25 * secondary_window_probs
        secondary_phase_probs = secondary_phase_probs / (tf.reduce_sum(secondary_phase_probs, axis=-1, keepdims=True) + 1e-6)
        secondary_insert_score = tf.reduce_sum(secondary_phase_probs[:, :, 1:], axis=-1, keepdims=True)
        secondary_phase_context = tf.einsum('bts,sd->btd', secondary_phase_probs, self.secondary_phase_token_matrix)
        secondary_context = self.secondary_insert_proj(tf.concat([
            main_anchor_bias,
            relation_context,
            secondary_phase_context,
            secondary_phase_probs,
            step_index_context,
        ], axis=-1)) * secondary_insert_score
        secondary_local_context = self.secondary_local_proj(tf.concat([
            x,
            main_anchor_bias,
            relation_context,
            secondary_phase_context,
            secondary_context,
            secondary_window_features,
        ], axis=-1))
        if not self.use_secondary_local_destination:
            secondary_context = tf.zeros_like(secondary_context)
            secondary_local_context = tf.zeros_like(secondary_local_context)
            secondary_insert_score = tf.zeros_like(secondary_insert_score)
        main_destination_candidate_field = tf.einsum('bz,zk->bk', main_destination_probs, self.origin_candidate_log_mask)
        main_destination_topk_candidate_field = tf.reduce_sum(main_dest_topk_weights[..., tf.newaxis] * tf.gather(self.origin_candidate_log_mask, main_dest_topk.indices), axis=-2)
        main_destination_candidate_field = 0.45 * main_destination_candidate_field + 0.55 * main_destination_topk_candidate_field
        home_candidate_field = tf.gather(self.origin_candidate_log_mask, home_zone_idx)
        work_candidate_field = tf.gather(self.origin_candidate_log_mask, work_zone_idx)
        origin_candidate_field = tf.gather(self.origin_candidate_log_mask, step_cat[:, :, 0])
        before_phase = secondary_phase_probs[:, :, 1:2]
        around_phase = secondary_phase_probs[:, :, 2:3]
        after_phase = secondary_phase_probs[:, :, 3:4]
        phase_destination_field = (
            before_phase * (0.65 * main_destination_candidate_field[:, tf.newaxis, :] + 0.35 * home_candidate_field[:, tf.newaxis, :]) +
            around_phase * (0.90 * main_destination_candidate_field[:, tf.newaxis, :] + 0.10 * origin_candidate_field) +
            after_phase * (0.65 * main_destination_candidate_field[:, tf.newaxis, :] + 0.25 * home_candidate_field[:, tf.newaxis, :] + 0.10 * work_candidate_field[:, tf.newaxis, :])
        )
        secondary_destination_gate = self.secondary_destination_gate(tf.concat([
            secondary_local_context,
            secondary_window_features,
            secondary_insert_score,
        ], axis=-1))
        learned_secondary_destination_bias = self.secondary_destination_bias_head(secondary_local_context)
        secondary_destination_query = self.secondary_destination_query_norm(
            self.secondary_destination_query(tf.concat([
                secondary_local_context,
                repeated_main_dest_context,
                relation_context,
                secondary_phase_context,
                step_index_context,
            ], axis=-1))
        )
        secondary_destination_local_logits = tf.einsum('btd,zd->btz', secondary_destination_query, zone_repr)
        secondary_destination_mix_gate = self.secondary_destination_mix_gate(tf.concat([
            secondary_local_context,
            repeated_main_dest_context,
            secondary_window_features,
            secondary_insert_score,
        ], axis=-1))
        secondary_destination_bias = self.secondary_destination_bias_scale * (
            secondary_destination_gate * learned_secondary_destination_bias +
            (1.0 - secondary_destination_gate) * self.secondary_phase_destination_scale * phase_destination_field
        ) * secondary_insert_score
        secondary_destination_local_logits = self.secondary_destination_local_scale * secondary_destination_mix_gate * secondary_destination_local_logits * secondary_insert_score
        if not self.use_secondary_local_destination:
            secondary_destination_bias = tf.zeros_like(secondary_destination_bias)
            secondary_destination_local_logits = tf.zeros_like(secondary_destination_local_logits)
        secondary_block_context = self.secondary_block_proj(tf.concat([
            secondary_local_context,
            repeated_main_purpose,
            repeated_main_dest_context,
            repeated_main_dest_purpose,
            secondary_phase_probs,
            secondary_window_features,
        ], axis=-1)) * secondary_insert_score

        purpose_context_raw = tf.concat([x, main_anchor_bias, relation_context, secondary_phase_context, secondary_context, secondary_local_context, secondary_block_context], axis=-1)
        purpose_context = self.purpose_utility_proj(purpose_context_raw) if self.use_interpretable_modules else self.purpose_plain_proj(purpose_context_raw)
        purpose_logits = self._mask_pad_class(self.purpose_head(purpose_context, training=training))
        purpose_logits += self.purpose_step_prior_scale * self.purpose_step_prior_matrix[tf.newaxis, :time_steps, :]
        first_step_mask = tf.one_hot(0, depth=time_steps, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
        work_purpose_bias = tf.one_hot(self.work_purpose_id, depth=self.num_purpose, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
        has_work_zone = tf.cast(work_zone_idx > 0, tf.float32)[:, tf.newaxis, tf.newaxis]
        purpose_logits += self.first_step_work_purpose_scale * first_step_mask * has_work_zone * work_purpose_bias
        purpose_logits += 0.20 * self.secondary_purpose_bias_head(secondary_local_context)
        purpose_logits += 0.18 * self.secondary_block_purpose_head(secondary_block_context)
        purpose_probs = tf.nn.softmax(purpose_logits, axis=-1)

        destination_semantic_raw = tf.concat([x, purpose_probs, main_anchor_bias, relation_context, secondary_phase_context, secondary_context, secondary_local_context, secondary_block_context], axis=-1)
        destination_semantic = self.destination_accessibility_proj(destination_semantic_raw) if self.use_interpretable_modules else self.destination_plain_proj(destination_semantic_raw)
        destination_query = self.destination_query_norm(self.destination_query(destination_semantic))
        destination_logits = tf.einsum('btd,zd->btz', destination_query, zone_repr)
        destination_logits += self.origin_destination_scale * tf.gather(self.origin_destination_prior_matrix, step_cat[:, :, 0])
        destination_logits += self.purpose_destination_scale * tf.einsum('btp,zp->btz', purpose_probs, self.zone_purpose_matrix)
        destination_logits += self.destination_support_scale * tf.einsum('btp,zp->btz', purpose_probs, self.zone_support_binary)
        destination_logits += self.main_anchor_conditioning_factor * self.main_destination_bias_scale * tf.einsum('bp,zp->bz', tf.cast(main_destination_purpose, destination_logits.dtype), self.zone_purpose_matrix)[:, tf.newaxis, :]
        destination_logits += self.main_anchor_conditioning_factor * secondary_destination_bias
        destination_logits += self.main_anchor_conditioning_factor * secondary_destination_local_logits
        destination_logits += self.origin_candidate_scale * tf.gather(self.origin_candidate_log_mask, step_cat[:, :, 0])
        home_one_hot = tf.one_hot(home_zone_idx, depth=self.num_zones, dtype=tf.float32)
        work_one_hot = tf.one_hot(work_zone_idx, depth=self.num_zones, dtype=tf.float32)
        destination_logits += self.home_destination_scale * purpose_probs[:, :, self.home_purpose_id:self.home_purpose_id + 1] * home_one_hot[:, tf.newaxis, :]
        destination_logits += self.work_destination_scale * purpose_probs[:, :, self.work_purpose_id:self.work_purpose_id + 1] * work_one_hot[:, tf.newaxis, :]
        destination_logits += self.first_step_work_destination_scale * first_step_mask * has_work_zone * work_one_hot[:, tf.newaxis, :]
        destination_logits = self._mask_pad_class(destination_logits)
        destination_probs = tf.nn.softmax(destination_logits, axis=-1)
        expected_zone_purpose_full = tf.einsum('btz,zp->btp', destination_probs, self.zone_purpose_matrix)
        expected_destination_context_full = tf.einsum('btz,zd->btd', destination_probs, zone_repr)
        topk = tf.math.top_k(destination_logits, k=min(self.destination_context_k, self.num_zones))
        topk_weights = tf.nn.softmax(topk.values, axis=-1)
        topk_zone_purpose = tf.gather(self.zone_purpose_matrix, topk.indices)
        topk_zone_repr = tf.gather(zone_repr, topk.indices)
        expected_zone_purpose_topk = tf.reduce_sum(topk_weights[..., tf.newaxis] * topk_zone_purpose, axis=-2)
        expected_destination_context_topk = tf.reduce_sum(topk_weights[..., tf.newaxis] * topk_zone_repr, axis=-2)
        alpha = tf.cast(self.destination_context_topk_alpha, tf.float32)
        expected_zone_purpose = (1.0 - alpha) * expected_zone_purpose_full + alpha * expected_zone_purpose_topk
        expected_destination_context = (1.0 - alpha) * expected_destination_context_full + alpha * expected_destination_context_topk

        time_values = tf.maximum(tf.cast(tf.range(self.num_time), tf.float32) - 1.0, 0.0)
        if not self.predict_depart_first:
            arrive_context_raw = tf.concat([x, purpose_probs, expected_zone_purpose, expected_destination_context, main_anchor_bias, relation_context, secondary_phase_context, secondary_context, secondary_local_context, secondary_block_context, step_index_context], axis=-1)
            arrive_context = self.scheduling_pressure_proj(arrive_context_raw) if self.use_interpretable_modules else self.arrive_plain_proj(arrive_context_raw)
            arrive_coarse_logits = self.arrive_coarse_head(arrive_context, training=training)
            arrive_logits = self._mask_pad_class(self.arrive_head(arrive_context, training=training))
            arrive_logits += self.arrive_step_prior_scale * self.arrive_step_prior_matrix[tf.newaxis, :time_steps, :]
            arrive_logits += self.purpose_arrive_prior_scale * tf.einsum('btp,pk->btk', purpose_probs, self.purpose_arrive_prior_matrix)
            arrive_logits += 0.20 * self.secondary_arrive_bias_head(secondary_local_context)
            arrive_logits += 0.15 * self.secondary_block_arrive_head(secondary_block_context)
            arrive_probs = tf.nn.softmax(arrive_logits, axis=-1)
            expected_arrive = tf.reduce_sum(arrive_probs * time_values[tf.newaxis, tf.newaxis, :], axis=-1)
            arrive_entropy = -tf.reduce_sum(arrive_probs * tf.math.log(tf.clip_by_value(arrive_probs, 1e-8, 1.0)), axis=-1)
            arrive_entropy = arrive_entropy / tf.math.log(tf.cast(tf.maximum(self.num_time, 2), tf.float32))
            arrive_top2 = tf.math.top_k(arrive_probs, k=min(2, self.num_time)).values
            if self.num_time > 1:
                arrive_top2_gap = arrive_top2[..., 0] - arrive_top2[..., 1]
            else:
                arrive_top2_gap = arrive_top2[..., 0]
            expected_arrive_norm = expected_arrive / tf.maximum(tf.cast(self.num_time - 2, tf.float32), 1.0)
            arrive_summary = tf.concat([arrive_probs, tf.stack([expected_arrive_norm], axis=-1)], axis=-1)
            arrive_depart_summary = tf.concat([
                arrive_context,
                tf.stack([expected_arrive_norm, arrive_entropy, arrive_top2_gap], axis=-1),
            ], axis=-1)
            arrive_hidden_summary = self.depart_arrive_summary_proj(arrive_depart_summary)
        else:
            batch_size = tf.shape(x)[0]
            arrive_coarse_logits = tf.zeros((batch_size, time_steps, 4), dtype=tf.float32)
            arrive_logits = tf.zeros((batch_size, time_steps, self.num_time), dtype=tf.float32)
            arrive_probs = tf.one_hot(tf.zeros((batch_size, time_steps), dtype=tf.int32), depth=self.num_time, dtype=tf.float32)
            expected_arrive = tf.zeros((batch_size, time_steps), dtype=tf.float32)
            arrive_entropy = tf.zeros((batch_size, time_steps), dtype=tf.float32)
            arrive_top2_gap = tf.zeros((batch_size, time_steps), dtype=tf.float32)
            expected_arrive_norm = tf.zeros((batch_size, time_steps), dtype=tf.float32)
            arrive_summary = tf.concat([arrive_probs, tf.stack([expected_arrive_norm], axis=-1)], axis=-1)
            arrive_hidden_summary = tf.zeros_like(x)

        origin_coords = tf.gather(self.zone_coord_matrix, step_cat[:, :, 0])
        expected_destination_coords = tf.einsum('btz,zc->btc', destination_probs, self.zone_coord_matrix)
        delta_lon = expected_destination_coords[:, :, 0] - origin_coords[:, :, 0]
        delta_lat = expected_destination_coords[:, :, 1] - origin_coords[:, :, 1]
        lat_mid_rad = (expected_destination_coords[:, :, 1] + origin_coords[:, :, 1]) * 0.5 * 0.01745329252
        lon_km = tf.abs(delta_lon) * 111.32 * tf.cos(lat_mid_rad)
        lat_km = tf.abs(delta_lat) * 110.57
        distance_est_km = tf.sqrt(tf.maximum(tf.square(lon_km) + tf.square(lat_km), 1e-6))
        short_trip = tf.nn.sigmoid((1.2 - distance_est_km) / 0.6)
        medium_trip = tf.nn.sigmoid((distance_est_km - 0.8) / 0.6) * tf.nn.sigmoid((6.0 - distance_est_km) / 1.2)
        long_trip = tf.nn.sigmoid((distance_est_km - 4.0) / 1.2)
        distance_features = self.mode_distance_proj(tf.stack([
            distance_est_km,
            tf.math.log1p(distance_est_km),
            tf.abs(delta_lon),
            tf.abs(delta_lat),
            short_trip,
            medium_trip,
            long_trip,
        ], axis=-1))

        distance_bin_weights = tf.stack([
            tf.cast(distance_est_km < 1.0, tf.float32),
            tf.cast((distance_est_km >= 1.0) & (distance_est_km < 3.0), tf.float32),
            tf.cast((distance_est_km >= 3.0) & (distance_est_km < 5.0), tf.float32),
            tf.cast((distance_est_km >= 5.0) & (distance_est_km < 10.0), tf.float32),
            tf.cast((distance_est_km >= 10.0) & (distance_est_km < 20.0), tf.float32),
            tf.cast(distance_est_km >= 20.0, tf.float32),
        ], axis=-1)

        if self.use_tour_resource_state:
            origin_is_home = tf.cast(step_cat[:, :, 0] == home_zone_idx[:, tf.newaxis], tf.float32)
            origin_is_work = tf.cast(step_cat[:, :, 0] == work_zone_idx[:, tf.newaxis], tf.float32)
            car_available_here, car_retrieval_needed, car_parked_dist_norm = self._scan_resource_state(step_cat[:, :, 0], step_cat[:, :, 2], home_zone_idx, self.car_mode_id)
            bike_available_here, bike_retrieval_needed, bike_parked_dist_norm = self._scan_resource_state(step_cat[:, :, 0], step_cat[:, :, 2], home_zone_idx, self.bike_mode_id)
            main_car_prob = repeated_main_mode[:, :, self.car_mode_id] if self.car_mode_id > 0 else tf.zeros_like(origin_is_home)
            main_bike_prob = repeated_main_mode[:, :, self.bike_mode_id] if self.bike_mode_id > 0 else tf.zeros_like(origin_is_home)
            resource_state = self.resource_state_proj(tf.concat([
                main_anchor_bias,
                relation_context,
                secondary_context,
                distance_features,
                tf.stack([
                    origin_is_home,
                    origin_is_work,
                    car_available_here,
                    bike_available_here,
                    car_retrieval_needed,
                    bike_retrieval_needed,
                    car_parked_dist_norm,
                    bike_parked_dist_norm,
                    main_car_prob,
                    main_bike_prob,
                ], axis=-1),
            ], axis=-1)) * self.resource_state_conditioning_factor
            resource_mode_bias = self.resource_state_conditioning_factor * self.resource_mode_bias_scale * self.resource_mode_bias_head(resource_state)
            if self.car_mode_id > 0:
                car_one_hot = tf.one_hot(self.car_mode_id, depth=self.num_mode, dtype=x.dtype)[tf.newaxis, tf.newaxis, :]
                resource_mode_bias += self.resource_state_conditioning_factor * ((0.8 * car_available_here - 1.1 * car_retrieval_needed)[:, :, tf.newaxis] * car_one_hot)
            if self.bike_mode_id > 0:
                bike_one_hot = tf.one_hot(self.bike_mode_id, depth=self.num_mode, dtype=x.dtype)[tf.newaxis, tf.newaxis, :]
                resource_mode_bias += self.resource_state_conditioning_factor * ((0.7 * bike_available_here - 0.9 * bike_retrieval_needed)[:, :, tf.newaxis] * bike_one_hot)
        else:
            resource_state = tf.zeros_like(x)
            resource_mode_bias = tf.zeros((tf.shape(x)[0], time_steps, self.num_mode), dtype=x.dtype)
            car_available_here = tf.zeros_like(seq_mask)
            bike_available_here = tf.zeros_like(seq_mask)
            car_retrieval_needed = tf.zeros_like(seq_mask)
            bike_retrieval_needed = tf.zeros_like(seq_mask)

        if self.predict_depart_first:
            depart_context = self.departure_adjustment_proj(tf.concat([
                x,
                purpose_probs,
                expected_zone_purpose,
                expected_destination_context,
                distance_features,
                main_anchor_bias,
                relation_context,
                secondary_phase_context,
                secondary_context,
                resource_state,
                step_index_context,
            ], axis=-1))
            depart_coarse_logits = self.depart_coarse_head(depart_context, training=training)
            direct_depart_logits = self._mask_pad_class(self.depart_head(depart_context, training=training))
            direct_depart_logits += self.depart_step_prior_scale * self.depart_step_prior_matrix[tf.newaxis, :time_steps, :]
            direct_depart_logits += self.purpose_depart_prior_scale * tf.einsum('btp,pk->btk', purpose_probs, self.purpose_depart_prior_matrix)
            depart_logits = direct_depart_logits
            depart_probs = tf.nn.softmax(depart_logits, axis=-1)
            expected_depart = tf.reduce_sum(depart_probs * time_values[tf.newaxis, tf.newaxis, :], axis=-1)
            expected_depart_norm = expected_depart / tf.maximum(tf.cast(self.num_time - 2, tf.float32), 1.0)
            depart_summary = tf.concat([depart_probs, tf.stack([expected_depart_norm], axis=-1)], axis=-1)

            mode_context = self.mode_feasibility_proj(tf.concat([
                x, purpose_probs, expected_zone_purpose, expected_destination_context, distance_features, depart_summary, main_anchor_bias, relation_context, secondary_phase_context, secondary_context, resource_state
            ], axis=-1))
            mode_family_logits = self.mode_family_head(mode_context, training=training)
            mode_family_probs = tf.nn.softmax(mode_family_logits, axis=-1)
            mode_context = tf.concat([mode_context, mode_family_probs], axis=-1)
            mode_logits = self._mask_pad_class(self.mode_head(mode_context, training=training))
            mode_logits += self.mode_step_prior_scale * self.mode_step_prior_matrix[tf.newaxis, :time_steps, :]
            mode_logits += self.mode_purpose_prior_scale * tf.einsum('btp,pm->btm', purpose_probs, self.purpose_mode_prior_matrix)
            mode_logits += self.mode_transition_prior_scale * tf.gather(self.mode_transition_prior_matrix, step_cat[:, :, 2])
            mode_logits += self.mode_usual_commute_prior_scale * tf.gather(self.mode_usual_commute_prior_matrix, static_cat[:, 4])[:, tf.newaxis, :]
            mode_logits += self.mode_distance_prior_scale * tf.einsum('btk,km->btm', distance_bin_weights, self.mode_distance_prior_matrix)
            mode_logits += self.mode_family_prior_scale * tf.einsum('btf,fm->btm', mode_family_probs, self.mode_family_to_mode_matrix)
            mode_logits += resource_mode_bias
            mode_probs = tf.nn.softmax(mode_logits, axis=-1)
            dwell_logits = tf.zeros((batch_size, time_steps, self.num_time), dtype=tf.float32)

            arrive_context = tf.concat([
                x,
                purpose_probs,
                expected_zone_purpose,
                expected_destination_context,
                distance_features,
                depart_summary,
                mode_probs,
                main_anchor_bias,
                secondary_phase_context,
                resource_state,
                step_index_context,
            ], axis=-1)
            arrive_coarse_logits = self.arrive_coarse_head(arrive_context, training=training)
            arrive_logits = self._mask_pad_class(self.arrive_head(arrive_context, training=training))
            arrive_logits += self.arrive_step_prior_scale * self.arrive_step_prior_matrix[tf.newaxis, :time_steps, :]
            arrive_logits += self.purpose_arrive_prior_scale * tf.einsum('btp,pk->btk', purpose_probs, self.purpose_arrive_prior_matrix)
            arrive_probs = tf.nn.softmax(arrive_logits, axis=-1)
            expected_arrive = tf.reduce_sum(arrive_probs * time_values[tf.newaxis, tf.newaxis, :], axis=-1)
            arrive_entropy = -tf.reduce_sum(arrive_probs * tf.math.log(tf.clip_by_value(arrive_probs, 1e-8, 1.0)), axis=-1)
            arrive_entropy = arrive_entropy / tf.math.log(tf.cast(tf.maximum(self.num_time, 2), tf.float32))
            arrive_top2 = tf.math.top_k(arrive_probs, k=min(2, self.num_time)).values
            if self.num_time > 1:
                arrive_top2_gap = arrive_top2[..., 0] - arrive_top2[..., 1]
            else:
                arrive_top2_gap = arrive_top2[..., 0]
            expected_arrive_norm = expected_arrive / tf.maximum(tf.cast(self.num_time - 2, tf.float32), 1.0)
            arrive_summary = tf.concat([arrive_probs, tf.stack([expected_arrive_norm], axis=-1)], axis=-1)
            duration_logits = self._mask_pad_class(self.duration_head(tf.concat([
                x, purpose_probs, expected_zone_purpose, expected_destination_context, mode_probs, distance_features, depart_summary, main_anchor_bias, relation_context, secondary_phase_context, secondary_context, resource_state, step_index_context
            ], axis=-1), training=training))
        else:
            mode_context_raw = tf.concat([
                x, purpose_probs, expected_zone_purpose, expected_destination_context, distance_features, arrive_summary, main_anchor_bias, relation_context, secondary_phase_context, secondary_context, resource_state
            ], axis=-1)
            mode_context = self.mode_feasibility_proj(mode_context_raw) if self.use_interpretable_modules else self.mode_plain_proj(mode_context_raw)
            mode_family_logits = self.mode_family_head(mode_context, training=training)
            mode_family_probs = tf.nn.softmax(mode_family_logits, axis=-1)
            mode_context = tf.concat([mode_context, mode_family_probs], axis=-1)
            mode_logits = self._mask_pad_class(self.mode_head(mode_context, training=training))
            mode_logits += self.mode_step_prior_scale * self.mode_step_prior_matrix[tf.newaxis, :time_steps, :]
            mode_logits += self.mode_purpose_prior_scale * tf.einsum('btp,pm->btm', purpose_probs, self.purpose_mode_prior_matrix)
            mode_logits += self.mode_transition_prior_scale * tf.gather(self.mode_transition_prior_matrix, step_cat[:, :, 2])
            mode_logits += self.mode_usual_commute_prior_scale * tf.gather(self.mode_usual_commute_prior_matrix, static_cat[:, 4])[:, tf.newaxis, :]
            mode_logits += self.mode_distance_prior_scale * tf.einsum('btk,km->btm', distance_bin_weights, self.mode_distance_prior_matrix)
            mode_logits += self.mode_family_prior_scale * tf.einsum('btf,fm->btm', mode_family_probs, self.mode_family_to_mode_matrix)
            mode_logits += resource_mode_bias
            mode_probs = tf.nn.softmax(mode_logits, axis=-1)

            dwell_context_raw = tf.concat([
                x,
                purpose_probs,
                expected_zone_purpose,
                expected_destination_context,
                arrive_summary,
                mode_probs,
                main_anchor_bias,
                relation_context,
                secondary_phase_context,
                secondary_context,
                secondary_local_context,
                resource_state,
                step_index_context,
            ], axis=-1)
            dwell_context = self.dwell_context_proj(dwell_context_raw) if self.use_interpretable_modules else self.dwell_plain_proj(dwell_context_raw)
            dwell_logits = self._mask_pad_class(self.dwell_head(dwell_context, training=training))
            dwell_probs = tf.nn.softmax(dwell_logits, axis=-1)
            expected_dwell = tf.reduce_sum(dwell_probs * time_values[tf.newaxis, tf.newaxis, :], axis=-1)
            expected_dwell_norm = expected_dwell / tf.maximum(tf.cast(self.num_time - 2, tf.float32), 1.0)
            dwell_summary = tf.concat([dwell_probs, tf.stack([expected_dwell_norm], axis=-1)], axis=-1)

            depart_arrive_gate = self.depart_arrive_gate(tf.concat([x, arrive_hidden_summary, step_index_context], axis=-1))
            depart_arrive_context = depart_arrive_gate * arrive_hidden_summary + (1.0 - depart_arrive_gate) * x
            depart_context_raw = tf.concat([
                x,
                purpose_probs,
                expected_zone_purpose,
                expected_destination_context,
                depart_arrive_context,
                tf.stack([expected_arrive_norm, arrive_entropy, arrive_top2_gap], axis=-1),
                mode_probs,
                dwell_summary,
                main_anchor_bias,
                relation_context,
                secondary_phase_context,
                secondary_context,
                resource_state,
                step_index_context,
            ], axis=-1)
            depart_context = self.departure_adjustment_proj(depart_context_raw) if self.use_interpretable_modules else self.depart_plain_proj(depart_context_raw)
            depart_coarse_logits = self.depart_coarse_head(depart_context, training=training)
            duration_logits = self._mask_pad_class(self.duration_head(tf.concat([
                x, purpose_probs, expected_zone_purpose, expected_destination_context, mode_probs, distance_features, main_anchor_bias, relation_context, secondary_phase_context, secondary_context, resource_state, step_index_context
            ], axis=-1), training=training))
            direct_depart_logits = self._mask_pad_class(self.depart_head(depart_context, training=training))
            direct_depart_logits += self.depart_step_prior_scale * self.depart_step_prior_matrix[tf.newaxis, :time_steps, :]
            direct_depart_logits += self.purpose_depart_prior_scale * tf.einsum('btp,pk->btk', purpose_probs, self.purpose_depart_prior_matrix)
            if self.use_depart_adjustment:
                dwell_bias_context = tf.concat([
                    depart_context,
                    dwell_context,
                    dwell_summary,
                    distance_features,
                    relation_context,
                    secondary_context,
                    secondary_local_context,
                    resource_state,
                ], axis=-1)
                direct_depart_logits += self.depart_dwell_bias_scale * self.depart_dwell_bias_head(dwell_bias_context, training=training)
            depart_logits = direct_depart_logits
        duration_probs = tf.nn.softmax(duration_logits, axis=-1)
        duration_depart_logits = direct_depart_logits
        depart_blend_gate = tf.ones((batch_size, time_steps, 1), dtype=x.dtype) * tf.cast(self.duration_depart_blend, x.dtype)
        if self.use_duration_depart and not self.predict_depart_first:
            depart_probs = tf.einsum('bta,btd,adk->btk', arrive_probs, duration_probs, self.depart_from_arrive_duration)
            depart_probs = depart_probs / (tf.reduce_sum(depart_probs, axis=-1, keepdims=True) + 1e-6)
            duration_depart_logits = tf.math.log(tf.clip_by_value(depart_probs, 1e-8, 1.0))
            if self.use_depart_adjustment:
                depart_blend_gate = self.depart_blend_gate_head(tf.concat([
                    depart_context,
                    dwell_context,
                    relation_context,
                    secondary_context,
                    step_index_context,
                ], axis=-1), training=training)
                blend = tf.cast(self.duration_depart_blend, x.dtype) * depart_blend_gate
            else:
                blend = tf.ones((batch_size, time_steps, 1), dtype=x.dtype) * tf.cast(self.duration_depart_blend, x.dtype)
            depart_logits = (1.0 - blend) * direct_depart_logits + blend * duration_depart_logits

        resource_violation_penalty = tf.constant(0.0, dtype=x.dtype)
        if self.use_tour_resource_state:
            denom = tf.reduce_sum(seq_mask) + 1e-6
            if self.car_mode_id > 0:
                resource_violation_penalty += tf.reduce_sum(mode_probs[:, :, self.car_mode_id] * car_retrieval_needed * seq_mask) / denom
            if self.bike_mode_id > 0:
                resource_violation_penalty += tf.reduce_sum(mode_probs[:, :, self.bike_mode_id] * bike_retrieval_needed * seq_mask) / denom

        outputs = {
            'purpose_logits': purpose_logits,
            'destination_logits': destination_logits,
            'arrive_logits': arrive_logits,
            'mode_family_logits': mode_family_logits,
            'mode_logits': mode_logits,
            'arrive_coarse_logits': arrive_coarse_logits,
            'depart_coarse_logits': depart_coarse_logits,
            'duration_logits': duration_logits,
            'dwell_logits': dwell_logits,
            'duration_depart_logits': duration_depart_logits,
            'depart_blend_gate': depart_blend_gate,
            'depart_logits': depart_logits,
            'main_step_logits': main_step_logits,
            'main_purpose_logits': main_purpose_logits,
            'main_destination_logits': main_destination_logits,
            'main_arrive_logits': main_arrive_logits,
            'main_mode_logits': main_mode_logits,
            'main_depart_logits': main_depart_logits,
            'relation_to_main_logits': relation_to_main_logits,
            'secondary_insert_logits': secondary_insert_logits,
            'resource_consistency_penalty': resource_violation_penalty,
        }
        if return_attention and attn_scores is not None:
            outputs['attention_scores'] = attn_scores
        return outputs






























