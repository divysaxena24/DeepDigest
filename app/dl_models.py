import tensorflow as tf
from tf_keras import layers, Model

class BahdanauAttentionTF(layers.Layer):
    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.W_q = layers.Dense(units)
        self.W_h = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, hidden_dim)
        # values shape: (batch_size, seq_len, hidden_dim)
        query_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W_q(query_time_axis) + self.W_h(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class AttentionSalienceScorerTF(Model):
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, gru_units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.bi_gru = layers.Bidirectional(
            layers.GRU(gru_units, return_sequences=True, return_state=True)
        )
        self.attention = BahdanauAttentionTF(units=gru_units)
        self.salience_head = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len)
        x = self.embedding(inputs)
        gru_out, forward_h, backward_h = self.bi_gru(x)
        state = tf.concat([forward_h, backward_h], axis=-1)
        context_vector, attention_weights = self.attention(state, gru_out)
        salience_score = self.salience_head(context_vector)
        return salience_score, attention_weights