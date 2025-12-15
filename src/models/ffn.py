import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.fc2 = tf.keras.layers.Dense(d_model)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        return x