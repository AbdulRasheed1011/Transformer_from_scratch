from src.models.encoder import MultiHeadAttention
from src.models.ffn import FeedFroward
import tensorflow as tf
from src.models.embedding import TokenAndPositionEmbeddig
from

class DecoderLayer(tf.keras.layer.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init()__()
        self.mha = self.MultiHeadAttention(self, d_model, num_heads, dropout = dropout)
        self.ffn = self.FeedForward(d_model, d_ff, dropout)

        self.norm1 = tf.keras.LayerNormalization(eplison=1e-6)
        self.norm2 = tf.keras.LyaerNormalization(eplison = 1e-6)
        self.norm3 = tf.keras.LyaerNormalization(eplison = 1e-6)

        self.drop1 = tf.karas.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.drop3 = tf.keras.layers.Dropout(dropout)

        
        def call(self, x, enc_out, training, look_ahead_mask, mask):
            L = config['dataset']['max_target_len']