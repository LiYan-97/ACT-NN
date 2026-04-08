from __future__ import annotations

from typing import Any

import tensorflow as tf


class BehaviorStructuredACTNN(tf.keras.Model):
    def __init__(
        self,
        num_zones: int,
        num_purpose: int,
        num_time: int,
        num_gap: int,
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
        home_return_step_prior,
        purpose_depart_prior_matrix,
        purpose_duration_prior_matrix,
        mode_duration_prior_matrix,
        first_depart_prior_matrix,
        gap_prior_matrix,
        duration_prior_matrix,
        continue_prior_matrix,
        continue_home_prior_matrix,
        mode_step_prior_matrix,
        purpose_mode_prior_matrix,
        mode_distance_prior_matrix,
        mode_transition_prior_matrix,
        mode_usual_commute_prior_matrix,
        home_purpose_id: int,
        bos_purpose_id: int,
        bos_mode_id: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(name='behavior_structured_actnn')
        self.num_zones = int(num_zones)
        self.num_purpose = int(num_purpose)
        self.num_time = int(num_time)
        self.num_gap = int(num_gap)
        self.num_mode = int(num_mode)
        self.home_purpose_id = int(home_purpose_id)
        self.bos_purpose_id = int(bos_purpose_id)
        self.bos_mode_id = int(bos_mode_id)
        self.hidden_dim = int(config.get('hidden_dim', 128))
        self.dropout_rate = float(config.get('dropout', 0.15))
        self.max_steps = int(config.get('max_steps', 16))
        self.use_step_embedding = bool(config.get('use_step_embedding', False))
        zone_embed_dim = int(config.get('zone_embed_dim', 32))
        zone_fixed_dim = int(config.get('zone_fixed_dim', 24))
        general_embed_dim = int(config.get('general_embed_dim', 8))

        self.zone_feature_matrix = tf.constant(zone_feature_matrix, dtype=tf.float32)
        self.zone_purpose_matrix = tf.constant(zone_purpose_matrix, dtype=tf.float32)
        self.zone_coord_matrix = tf.constant(zone_coord_matrix, dtype=tf.float32)
        self.purpose_step_prior_matrix = tf.constant(purpose_step_prior_matrix, dtype=tf.float32)
        self.home_return_step_prior = tf.constant(home_return_step_prior, dtype=tf.float32)
        self.purpose_depart_prior_matrix = tf.constant(purpose_depart_prior_matrix, dtype=tf.float32)
        self.purpose_duration_prior_matrix = tf.constant(purpose_duration_prior_matrix, dtype=tf.float32)
        self.mode_duration_prior_matrix = tf.constant(mode_duration_prior_matrix, dtype=tf.float32)
        self.first_depart_prior_matrix = tf.constant(first_depart_prior_matrix, dtype=tf.float32)
        self.gap_prior_matrix = tf.constant(gap_prior_matrix, dtype=tf.float32)
        self.duration_prior_matrix = tf.constant(duration_prior_matrix, dtype=tf.float32)
        self.continue_prior_matrix = tf.constant(continue_prior_matrix, dtype=tf.float32)
        self.continue_home_prior_matrix = tf.constant(continue_home_prior_matrix, dtype=tf.float32)
        self.mode_step_prior_matrix = tf.constant(mode_step_prior_matrix, dtype=tf.float32)
        self.purpose_mode_prior_matrix = tf.constant(purpose_mode_prior_matrix, dtype=tf.float32)
        self.mode_distance_prior_matrix = tf.constant(mode_distance_prior_matrix, dtype=tf.float32)
        self.mode_transition_prior_matrix = tf.constant(mode_transition_prior_matrix, dtype=tf.float32)
        self.mode_usual_commute_prior_matrix = tf.constant(mode_usual_commute_prior_matrix, dtype=tf.float32)
        self.mode_family_membership = tf.constant(self._build_mode_family_membership(config), dtype=tf.float32)
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
        self.static_context = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            ],
            name='static_context',
        )
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
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim * 2, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.hidden_dim),
            ],
            name='ffn',
        )
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ffn_norm')

        self.purpose_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_purpose),
            ],
            name='purpose_head',
        )
        self.first_depart_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_time),
            ],
            name='first_depart_head',
        )
        self.gap_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_gap),
            ],
            name='gap_head',
        )
        self.duration_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_time),
            ],
            name='duration_head',
        )
        self.mode_family_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(4),
            ],
            name='mode_family_head',
        )
        self.mode_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_mode),
            ],
            name='mode_head',
        )
        self.continue_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(2),
            ],
            name='continue_head',
        )
        self.destination_query = tf.keras.layers.Dense(zone_embed_dim + zone_fixed_dim, name='destination_query')
        self.destination_query_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='destination_query_norm')
        self.mode_context_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='mode_context_proj')
        self.mode_distance_proj = tf.keras.layers.Dense(16, activation='relu', name='mode_distance_proj')
        self.duration_context_proj = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='duration_context_proj')
        self.purpose_compatibility_scale = self.add_weight(
            name='purpose_compatibility_scale', shape=(), initializer=tf.keras.initializers.Constant(0.6), trainable=True
        )
        self.home_destination_scale = self.add_weight(
            name='home_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True
        )
        self.closure_purpose_scale = self.add_weight(
            name='closure_purpose_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('closure_purpose_init', 1.2))), trainable=True
        )
        self.closure_destination_scale = self.add_weight(
            name='closure_destination_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('closure_destination_init', 1.0))), trainable=True
        )
        self.purpose_step_prior_scale = self.add_weight(
            name='purpose_step_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('purpose_step_prior_init', 1.0))), trainable=True
        )
        self.home_return_step_prior_scale = self.add_weight(
            name='home_return_step_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('home_return_step_prior_init', 1.2))), trainable=True
        )
        self.purpose_depart_prior_scale = self.add_weight(
            name='purpose_depart_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('purpose_depart_prior_init', 0.8))), trainable=True
        )
        self.purpose_duration_prior_scale = self.add_weight(
            name='purpose_duration_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('purpose_duration_prior_init', 0.8))), trainable=True
        )
        self.mode_duration_prior_scale = self.add_weight(
            name='mode_duration_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('mode_duration_prior_init', 0.8))), trainable=True
        )
        prior_init = float(config.get('time_prior_init', 0.8))
        continue_prior_init = float(config.get('continue_prior_init', 1.2))
        continue_behavior_init = float(config.get('continue_behavior_init', 1.0))
        continue_home_prior_init = float(config.get('continue_home_prior_init', 2.0))
        mode_prior_init = float(config.get('mode_prior_init', 1.1))
        mode_distance_prior_init = float(config.get('mode_distance_prior_init', 1.8))
        self.first_depart_prior_scale = self.add_weight(
            name='first_depart_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(prior_init), trainable=True
        )
        self.gap_prior_scale = self.add_weight(
            name='gap_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(prior_init), trainable=True
        )
        self.duration_prior_scale = self.add_weight(
            name='duration_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(prior_init), trainable=True
        )
        self.continue_prior_scale = self.add_weight(
            name='continue_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(continue_prior_init), trainable=True
        )
        self.continue_behavior_scale = self.add_weight(
            name='continue_behavior_scale', shape=(), initializer=tf.keras.initializers.Constant(continue_behavior_init), trainable=True
        )
        self.continue_home_prior_scale = self.add_weight(
            name='continue_home_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(continue_home_prior_init), trainable=True
        )
        self.mode_step_prior_scale = self.add_weight(
            name='mode_step_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(mode_prior_init), trainable=True
        )
        self.mode_purpose_prior_scale = self.add_weight(
            name='mode_purpose_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(mode_prior_init), trainable=True
        )
        self.mode_distance_prior_scale = self.add_weight(
            name='mode_distance_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(mode_distance_prior_init), trainable=True
        )
        self.mode_transition_prior_scale = self.add_weight(
            name='mode_transition_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('mode_transition_prior_init', 0.9))), trainable=True
        )
        self.mode_usual_commute_prior_scale = self.add_weight(
            name='mode_usual_commute_prior_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('mode_usual_commute_prior_init', 0.9))), trainable=True
        )
        self.mode_family_scale = self.add_weight(
            name='mode_family_scale', shape=(), initializer=tf.keras.initializers.Constant(float(config.get('mode_family_init', 1.0))), trainable=True
        )

    def _mask_pad_class(self, logits: tf.Tensor) -> tf.Tensor:
        neg = tf.cast(-1e9, logits.dtype)
        masked_first = tf.fill(tf.shape(logits[..., :1]), neg)
        return tf.concat([masked_first, logits[..., 1:]], axis=-1)

    def _zone_context(self, zone_idx: tf.Tensor) -> tf.Tensor:
        trainable_embed = self.zone_embedding(zone_idx)
        fixed_features = tf.gather(self.zone_feature_matrix, zone_idx)
        fixed_embed = self.zone_feature_proj(fixed_features)
        return tf.concat([trainable_embed, fixed_embed], axis=-1)

    def _build_mode_family_membership(self, config: dict[str, Any]):
        mat = [[0.0] * self.num_mode for _ in range(4)]
        for family_idx, ids in enumerate([
            [int(config.get('walk_mode_id', -1)), int(config.get('bike_mode_id', -1))],
            [int(config.get('bus_mode_id', -1)), int(config.get('metro_mode_id', -1))],
            [int(config.get('taxi_mode_id', -1)), int(config.get('car_mode_id', -1))],
            [self.num_mode - 1],
        ]):
            for mode_id in ids:
                if 0 < mode_id < self.num_mode:
                    mat[family_idx][mode_id] = 1.0
        return mat

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

        x = tf.concat(
            [
                static_context_tiled,
                origin_zone_context,
                prev_purpose_context,
                prev_mode_context,
                district_context,
                step_num_context,
            ],
            axis=-1,
        )
        x = self.sequence_input_proj(x)
        x = self.sequence_dropout(x, training=training)
        x = self.encoder(x, mask=tf.cast(seq_mask > 0, tf.bool), training=training)

        if return_attention:
            attn_out, attn_scores = self.self_attention(
                x,
                x,
                attention_mask=self._attention_mask(seq_mask),
                return_attention_scores=True,
                training=training,
            )
        else:
            attn_out = self.self_attention(x, x, attention_mask=self._attention_mask(seq_mask), training=training)
            attn_scores = None
        x = self.attention_norm(x + attn_out)
        x = self.ffn_norm(x + self.ffn(x, training=training))
        x = x * seq_mask[:, :, tf.newaxis]

        purpose_logits = self._mask_pad_class(self.purpose_head(x, training=training))
        purpose_logits += self.purpose_step_prior_scale * self.purpose_step_prior_matrix[tf.newaxis, :time_steps, :]
        closure_score = step_num[:, :, 0:1] * (1.0 - step_num[:, :, 3:4])
        home_purpose_bias = self.closure_purpose_scale * closure_score * tf.one_hot(self.home_purpose_id, depth=self.num_purpose, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
        purpose_logits += home_purpose_bias
        purpose_probs = tf.nn.softmax(purpose_logits, axis=-1)

        zone_repr = self._zone_context(tf.range(self.num_zones, dtype=tf.int32))
        destination_query = self.destination_query_norm(self.destination_query(tf.concat([x, purpose_probs], axis=-1)))
        destination_logits = tf.einsum('btd,zd->btz', destination_query, zone_repr)
        destination_logits += self.purpose_compatibility_scale * tf.einsum('btp,zp->btz', purpose_probs, self.zone_purpose_matrix)
        home_bias = self.home_destination_scale * purpose_probs[:, :, self.home_purpose_id : self.home_purpose_id + 1]
        home_bias += self.home_return_step_prior_scale * self.home_return_step_prior[tf.newaxis, :time_steps, tf.newaxis]
        home_bias += self.closure_destination_scale * closure_score
        home_one_hot = tf.one_hot(home_zone_idx, depth=self.num_zones, dtype=tf.float32)
        destination_logits += home_bias * home_one_hot[:, tf.newaxis, :]
        destination_logits = self._mask_pad_class(destination_logits)
        destination_probs = tf.nn.softmax(destination_logits, axis=-1)
        expected_zone_purpose = tf.einsum('btz,zp->btp', destination_probs, self.zone_purpose_matrix)
        expected_destination_context = tf.einsum('btz,zd->btd', destination_probs, zone_repr)

        depart_context = tf.concat([x, purpose_probs, expected_zone_purpose, expected_destination_context, step_index_context], axis=-1)
        first_depart_logits = self._mask_pad_class(self.first_depart_head(depart_context, training=training))
        gap_logits = self._mask_pad_class(self.gap_head(depart_context, training=training))
        first_depart_logits += self.first_depart_prior_scale * self.first_depart_prior_matrix[tf.newaxis, :time_steps, :]
        first_depart_logits += self.purpose_depart_prior_scale * tf.einsum('btp,pk->btk', purpose_probs, self.purpose_depart_prior_matrix)
        gap_logits += self.gap_prior_scale * self.gap_prior_matrix[tf.newaxis, :time_steps, :]

        first_depart_probs = tf.nn.softmax(first_depart_logits, axis=-1)
        gap_probs = tf.nn.softmax(gap_logits, axis=-1)
        time_values = tf.maximum(tf.cast(tf.range(self.num_time), tf.float32) - 1.0, 0.0)
        gap_values = tf.concat([tf.zeros((1,), dtype=tf.float32), tf.range(-2.0, 9.0, dtype=tf.float32)], axis=0)
        expected_first_depart = tf.reduce_sum(first_depart_probs * time_values[tf.newaxis, tf.newaxis, :], axis=-1)
        expected_gap = tf.reduce_sum(gap_probs * gap_values[tf.newaxis, tf.newaxis, :], axis=-1)
        time_summary = tf.stack([expected_first_depart, expected_gap], axis=-1)

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
        distance_features = self.mode_distance_proj(
            tf.stack([distance_est_km, tf.math.log1p(distance_est_km), tf.abs(delta_lon), tf.abs(delta_lat), short_trip, medium_trip, long_trip], axis=-1)
        )

        mode_context = self.mode_context_proj(
            tf.concat([x, purpose_probs, expected_zone_purpose, expected_destination_context, distance_features, time_summary], axis=-1)
        )
        mode_family_logits = self.mode_family_head(mode_context, training=training)
        mode_family_probs = tf.nn.softmax(mode_family_logits, axis=-1)
        mode_logits = self._mask_pad_class(self.mode_head(mode_context, training=training))
        mode_logits += self.mode_family_scale * tf.einsum('btf,fm->btm', mode_family_probs, self.mode_family_membership)
        mode_logits += self.mode_step_prior_scale * self.mode_step_prior_matrix[tf.newaxis, :time_steps, :]
        mode_logits += self.mode_purpose_prior_scale * tf.einsum('btp,pm->btm', purpose_probs, self.purpose_mode_prior_matrix)
        mode_logits += self.mode_transition_prior_scale * tf.gather(self.mode_transition_prior_matrix, step_cat[:, :, 2])
        mode_logits += self.mode_usual_commute_prior_scale * tf.gather(self.mode_usual_commute_prior_matrix, static_cat[:, 4])[:, tf.newaxis, :]

        distance_bin_weights = tf.stack([
            tf.cast(distance_est_km < 1.0, tf.float32),
            tf.cast((distance_est_km >= 1.0) & (distance_est_km < 3.0), tf.float32),
            tf.cast((distance_est_km >= 3.0) & (distance_est_km < 5.0), tf.float32),
            tf.cast((distance_est_km >= 5.0) & (distance_est_km < 10.0), tf.float32),
            tf.cast((distance_est_km >= 10.0) & (distance_est_km < 20.0), tf.float32),
            tf.cast(distance_est_km >= 20.0, tf.float32),
        ], axis=-1)
        mode_logits += self.mode_distance_prior_scale * tf.einsum('btk,km->btm', distance_bin_weights, self.mode_distance_prior_matrix)
        mode_probs = tf.nn.softmax(mode_logits, axis=-1)

        duration_context = self.duration_context_proj(
            tf.concat([x, purpose_probs, expected_zone_purpose, expected_destination_context, mode_probs, time_summary, step_index_context], axis=-1)
        )
        duration_logits = self._mask_pad_class(self.duration_head(duration_context, training=training))
        duration_logits += self.duration_prior_scale * self.duration_prior_matrix[tf.newaxis, :time_steps, :]
        duration_logits += self.purpose_duration_prior_scale * tf.einsum('btp,pk->btk', purpose_probs, self.purpose_duration_prior_matrix)
        duration_logits += self.mode_duration_prior_scale * tf.einsum('btm,mk->btk', mode_probs, self.mode_duration_prior_matrix)
        duration_probs = tf.nn.softmax(duration_logits, axis=-1)
        expected_duration = tf.reduce_sum(duration_probs * time_values[tf.newaxis, tf.newaxis, :], axis=-1)
        late_score = tf.nn.sigmoid((((expected_first_depart + expected_duration) / tf.maximum(tf.cast(self.num_time - 2, tf.float32), 1.0)) - 0.7) / 0.08)
        purpose_home_prob = purpose_probs[:, :, self.home_purpose_id]
        home_return_prob = tf.reduce_sum(destination_probs * home_one_hot[:, tf.newaxis, :], axis=-1)
        continue_features = tf.stack([home_return_prob, purpose_home_prob, late_score], axis=-1)

        continue_logits = self.continue_head(
            tf.concat([x, purpose_probs, expected_zone_purpose, mode_probs, time_summary, continue_features, step_index_context], axis=-1), training=training
        )
        continue_logits += self.continue_prior_scale * self.continue_prior_matrix[tf.newaxis, :time_steps, :]
        continue_home_prior = (
            (1.0 - home_return_prob)[..., tf.newaxis] * self.continue_home_prior_matrix[tf.newaxis, :time_steps, 0, :]
            + home_return_prob[..., tf.newaxis] * self.continue_home_prior_matrix[tf.newaxis, :time_steps, 1, :]
        )
        continue_logits += self.continue_home_prior_scale * continue_home_prior
        stop_bias = 1.5 * home_return_prob * purpose_home_prob + 0.8 * late_score
        continue_bias = 0.6 * (1.0 - home_return_prob) * (1.0 - purpose_home_prob)
        continue_logits += self.continue_behavior_scale * tf.stack([stop_bias, continue_bias], axis=-1)

        outputs = {
            'purpose_logits': purpose_logits,
            'destination_logits': destination_logits,
            'first_depart_logits': first_depart_logits,
            'gap_logits': gap_logits,
            'duration_logits': duration_logits,
            'mode_family_logits': mode_family_logits,
            'mode_logits': mode_logits,
            'continue_logits': continue_logits,
            'purpose_probs': purpose_probs,
            'destination_probs': destination_probs,
        }
        if return_attention and attn_scores is not None:
            outputs['attention_scores'] = attn_scores
        return outputs








