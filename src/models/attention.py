import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (B, H, Lq, Dh)
    k: (B, H, Lk, Dh)
    v: (B, H, Lk, Dh)
    mask:
      - encoder padding mask: (B,1,1,Lk)
      - decoder look-ahead+pad: (B,1,Lq,Lk)

    returns:
      out: (B, H, Lq, Dh)
      attn_weights: (B, H, Lq, Lk)
    """
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)  # (B,H,Lq,Lk)

    if mask is not None:
        # mask is 1 keep, 0 block
        scores += (1.0 - tf.cast(mask, tf.float32)) * (-1e9)

    attn_weights = tf.nn.softmax(scores, axis=-1)                  # (B,H,Lq,Lk)
    out = tf.matmul(attn_weights, v)                               # (B,H,Lq,Dh)
    return out, attn_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False)

        self.wo = tf.keras.layers.Dense(d_model, use_bias=False)
        self.drop = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        x = tf.reshape(x, (B, L, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])  # (B,H,L,Dh)

    def _combine_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])     # (B,L,H,Dh)
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        return tf.reshape(x, (B, L, self.d_model))

    def call(self, q_in, k_in, v_in, mask=None, training=False):
        """
        General MHA:
          q_in: (B, Lq, d_model)
          k_in: (B, Lk, d_model)
          v_in: (B, Lk, d_model)
        """
        q = self._split_heads(self.wq(q_in))  # (B,H,Lq,Dh)
        k = self._split_heads(self.wk(k_in))  # (B,H,Lk,Dh)
        v = self._split_heads(self.wv(v_in))  # (B,H,Lk,Dh)

        out, attn_w = scaled_dot_product_attention(q, k, v, mask=mask)
        out = self._combine_heads(out)        # (B,Lq,d_model)
        out = self.wo(out)
        out = self.drop(out, training=training)
        return out, attn_w