import tensorflow as tf
from src.models.attention import MultiHeadAttention
from src.models.ffn import FeedForward

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.drop3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_out, look_ahead_mask=None, enc_pad_mask=None, training=False):
        """
        x: (B, T, d_model)         decoder input embeddings
        enc_out: (B, S, d_model)   encoder output
        look_ahead_mask: (B,1,T,T) combined mask for decoder self-attn
        enc_pad_mask: (B,1,1,S)    padding mask for encoder outputs
        """

        # 1) Masked self-attention (decoder attends leftward only)
        attn1, attn_w1 = self.self_mha(
            q_in=x, k_in=x, v_in=x,
            mask=look_ahead_mask,
            training=training
        )
        x = self.norm1(x + self.drop1(attn1, training=training))

        # 2) Cross-attention (decoder attends to encoder output)
        attn2, attn_w2 = self.cross_mha(
            q_in=x, k_in=enc_out, v_in=enc_out,
            mask=enc_pad_mask,
            training=training
        )
        x = self.norm2(x + self.drop2(attn2, training=training))

        # 3) Feed-forward
        ffn_out = self.ffn(x, training=training)
        x = self.norm3(x + self.drop3(ffn_out, training=training))

        return x, attn_w1, attn_w2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ]
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_out, look_ahead_mask=None, enc_pad_mask=None, training=False):
        x = self.drop(x, training=training)

        attn = {}
        for i, layer in enumerate(self.layers):
            x, attn_w1, attn_w2 = layer(
                x, enc_out,
                look_ahead_mask=look_ahead_mask,
                enc_pad_mask=enc_pad_mask,
                training=training
            )
            attn[f"decoder_layer_{i+1}_self"] = attn_w1     # (B,H,T,T)
            attn[f"decoder_layer_{i+1}_cross"] = attn_w2    # (B,H,T,S)

        return x, attn