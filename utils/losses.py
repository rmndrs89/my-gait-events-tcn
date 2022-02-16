import tensorflow as tf
import tensorflow.keras.backend as K

class MyWeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights=0.01):
        super().__init__()
        self.weights = weights
    
    def call(self, y_true, y_pred):
        loss = K.mean(K.binary_crossentropy(y_true, y_pred) * (y_true + self.weights), axis=-1)
        return loss

class MyWeigtedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights=0.01):
        super().__init__()
        self.weights = weights
    
    def call(self, y_true, y_pred):
        y_hat = tf.cast(y_true>=1, tf.float32)
        loss = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred) * (y_hat + self.weights), axis=-1)
        return loss