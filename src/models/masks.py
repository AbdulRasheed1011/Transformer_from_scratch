import tensorflow as tf

def create_padding_mask(input_ids, pad_token_id):
    mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), tf.float32)  #(batch, seq_len)
    return mask[:,tf.newaxis, tf.newaxis, :]  # (batch, 1, 1, seq_len)