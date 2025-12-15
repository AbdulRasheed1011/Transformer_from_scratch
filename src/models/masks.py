import tensorflow as tf

def create_padding_mask(input_ids, pad_token_id):
    mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), tf.float32)  #(batch, seq_len)
    return mask[:,tf.newaxis, tf.newaxis, :]  # (batch, 1, 1, seq_len)

def create_look_ahead_mask(seq_len):
    ones = tf.ones(seq_len, seq_len), tf.float32)
    mask = tf.linalg.band_part(ones, -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]

def combine_masks(padding_mask, look_ahead_mask):
    pm = padding_mask[:,:,:,:]
    pm = tf.broadcast_to(pm, (tf.shape(pm)[0], 1, tf.shape(look_ahead_mask)[2], tf.shape(pm)[-1])) #(B, 1, L, L)
    lam = tf.broadcast_to(look_ahead_mask, (tf.shape(pm)[0], 1, tf.shape(look_ahead_mask[2], tf.shape(look_ahead_mask)[3])))
    return pm*lam