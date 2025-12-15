import tensorflow as tf
from src.models.attention import MultiHeadAttention
from src.models.ffn import FeedForward

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        # 1) Self-attention sublayer
        attn_out, attn_w = self.mha(x, mask=mask, training=training)   # (B,L,d_model)
        x = self.norm1(x + self.drop1(attn_out, training=training))    # residual + norm

        # 2) Feed-forward sublayer
        ffn_out = self.ffn(x, training=training)                       # (B,L,d_model)
        x = self.norm2(x + self.drop2(ffn_out, training=training))     # residual + norm

        return x, attn_w


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ]
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None, training=False): 
        x = self.drop(x, training=training)
        attention_weights = {}

        for i, layer in enumerate(self.layers):
            x, attn_w = layer(x, mask=mask, training=training)
            attention_weights[f"encoder_layer_{i+1}"] = attn_w

        return x, attention_weights