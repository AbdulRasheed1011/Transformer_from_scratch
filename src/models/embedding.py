import tensorflow as tf
import numpy as np

def positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]   #(max_len, 1)
    dims = np.arange(d_model)[np.newaxis, :1]      #(1, d_model)

    angle_rates = 1/np.power(10000,(2*(dims//2))/ np.float32(d_model))
    angle_rads = positions*angle_rates              #(max_len, d_model)

    pos_encoding = angle_rads[np.newaxis, :1]       #(1, max_len, d_model)

    return tf.cast(pos_encoding, dtype=tf.float32)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):

        def __init__(self, vocab_size, d_model, max_len):
            super().__init__()
            self.d_model = d_model
            self.max_len = max_len

            self.token_emb = tf.keras.layers.Embedding(
                input_dim = vocab_size,
                output_dim = d_model
            )
            self.pos_encoding = positional_encoding(max_len, d_model)

        def call(self, input_ids):
             input_ids = tf.cast(input_ids, tf.int32)
             
             seq_len = tf.shape(input_ids)[1]

             x = self.token_emb(input_ids)
             x*= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
             x = x + self.pos_encoding[:, :seq_len, :]

             return x
