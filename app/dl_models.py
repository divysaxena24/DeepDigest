import tensorflow as tf
from tensorflow.keras import layers, Model

class BahdanauAttentionTF(layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W_q = layers.Dense(units, use_bias=False)
        self.W_k = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)

    def call(self, query, values):
        query_expanded = tf.expand_dims(query, axis=1)
        score = self.V(tf.nn.tanh(self.W_q(query_expanded) + self.W_k(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attention_weights, axis=-1)

class AttentionSalienceScorerTF(Model):
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.bi_gru = layers.Bidirectional(
            layers.GRU(hidden_dim, return_sequences=True, return_state=True)
        )
        self.attention = BahdanauAttentionTF(hidden_dim * 2)
        self.classifier = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

    def call(self, inputs, training=False):
        gru_out, state_f, state_b = self.bi_gru(inputs)
        query = tf.concat([state_f, state_b], axis=-1)
        context_vector, attn_weights = self.attention(query, gru_out)
        chunk_scores = tf.squeeze(self.classifier(gru_out), axis=-1)
        return chunk_scores, attn_weights