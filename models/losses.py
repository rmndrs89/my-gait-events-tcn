import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class MyWeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights=0.01):
        super().__init__()
        self.weights = weights
    
    def call(self, y_true, y_pred):
        loss = K.mean(K.binary_crossentropy(y_true, y_pred) * (y_true + self.weights), axis=-1)
        return loss

class MyWeightedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, weights=0.01):
        super().__init__()
        self.weights = weights
    
    def call(self, y_true, y_pred):
        loss = K.mean(K.mean(K.square(y_pred - y_true)) * (y_true + self.weights), axis=-1)
        return loss