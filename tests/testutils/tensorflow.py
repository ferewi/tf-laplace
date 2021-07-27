from unittest import mock
import tensorflow as tf


class RealTfModel:

    def __init__(self, model):
        self.model = model
        self.input = tf.ones([1, 2]) * 1
        self.y_true = [[9.]]
        self.loss = tf.keras.losses.MeanSquaredError()

    @classmethod
    def create(cls):
        ones_init = tf.keras.initializers.ones
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(3, input_dim=2, activation='linear', kernel_initializer=ones_init,
                                  bias_initializer=ones_init),
            tf.keras.layers.Dense(2, activation='linear', kernel_initializer=ones_init, bias_initializer=ones_init),
        ])
        return cls(model)

    def get(self):
        return self.model

    def backward(self):
        with tf.GradientTape() as tape:
            logits = self.model(self.input)
            loss_val = self.loss(logits, self.y_true)
            grads = tape.gradient(loss_val, self.model.trainable_weights)
            return grads


class ModelMocker:

    @staticmethod
    def mock_layer(name, shape):
        layer = mock.create_autospec(tf.keras.layers.Dense)
        layer.name = name
        kernel_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = name + '/kernel:0'
        kernel_weights.shape = shape
        bias_weights = mock.create_autospec(tf.Variable)
        bias_weights.name = name + '/bias:0'
        bias_weights.shape = [shape[1]]
        layer.weights = [kernel_weights, bias_weights]
        return layer

    @staticmethod
    def mock_model():
        model = mock.create_autospec(tf.keras.models.Sequential)
        return model