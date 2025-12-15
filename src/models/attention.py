import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (B, H, Lq, Dh)
    k: (B, H, Lk, Dh)
    v: (B, H, Lk, Dh)
    mask: (B, 1, 1, Lk) with 1 for keep, 0 for pad (optional)

    returns:
      out: (B, H, Lq, Dh)
      attn_weights: (B, H, Lq, Lk)
    """
    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    # (B,H,Lq,Lk)
    scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)

    # If mask is 1 for real tokens, convert it to an additive mask for logits:
    # where pad (0) becomes -1e9 (so softmax ~ 0)
    if mask is not None:
        # mask: (B,1,1,Lk) -> broadcast to (B,H,Lq,Lk)
        scores += (1.0 - tf.cast(mask, tf.float32)) * (-1e9)

    attn_weights = tf.nn.softmax(scores, axis=-1)          # (B,H,Lq,Lk)
    out = tf.matmul(attn_weights, v)                       # (B,H,Lq,Dh)
    return out, attn_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear projections
        self.wq = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)

        self.wo = tf.keras.layers.Dense(d_model, use_bias=True)
        self.drop = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x):
        """
        x: (B, L, d_model)
        -> (B, H, L, Dh)
        """
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        x = tf.reshape(x, (B, L, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        """
        x: (B, H, L, Dh)
        -> (B, L, d_model)
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, L, H, Dh)
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        return tf.reshape(x, (B, L, self.d_model))

    def call(self, x, mask=None, training=False):
        """
        Self-attention:
        x: (B, L, d_model)
        mask: (B, 1, 1, L) optional
        """
        q = self._split_heads(self.wq(x))   # (B,H,L,Dh)
        k = self._split_heads(self.wk(x))   # (B,H,L,Dh)
        v = self._split_heads(self.wv(x))   # (B,H,L,Dh)

        attn_out, attn_w = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_out = self._combine_heads(attn_out)           # (B,L,d_model)
        attn_out = self.wo(attn_out)                       # (B,L,d_model)
        attn_out = self.drop(attn_out, training=training)

        return attn_out, attn_w