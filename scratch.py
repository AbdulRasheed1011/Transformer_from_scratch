import tensorflow as tf
from src.models.embedding import TokenAndPositionEmbedding

batch_size = 32
seq_len = 512
vocab_size = 32000
d_model = 512
max_len = 512

fake_ids = tf.random.uniform(
    shape=(batch_size, seq_len),
    minval=0,
    maxval=vocab_size,
    dtype=tf.int32,             # or int64; we cast inside the layer
)

embed_layer = TokenAndPositionEmbedding(vocab_size, d_model, max_len)
x = embed_layer(fake_ids)
print(x.shape)  # should be (32, 512, 512)