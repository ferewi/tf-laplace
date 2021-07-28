from abc import ABC, abstractmethod
import tensorflow as tf
from laplace.curvature import Curvature


class Sampler(ABC):

    def __init__(self, curvature: Curvature, tau: float, n: float):
        self._curvature = curvature
        self._curvature.invert(tau, n)
        self.norm = tau
        self.scale = n

    @staticmethod
    def create(curvature: Curvature, tau: float, n: float):
        sampler_by_curvature = {
            "DiagFisher": DiagFisherSampler,
            "BlockDiagFisher": BlockDiagFisherSampler,
            "KFAC": KFACSampler
        }
        return sampler_by_curvature[curvature.__class__.__name__](curvature, tau, n)

    @abstractmethod
    def sample_and_replace_weights(self):
        pass

    def _split_kernel_bias_weights_and_replace(self, layer: tf.keras.layers.Layer, x: tf.Tensor):
        if len(layer.weights) == 1:
            x_k, x_b = x, tf.zeros(shape=(), dtype=tf.float32)
            pass
        elif len(layer.weights) == 2:
            x_k, x_b = tf.split(x, [x.shape[0] - 1, 1], axis=0)
            pass
        else:
            raise RuntimeError(f"Can't determine kernel/bias weights split for layer {layer},"
                               f" having {len(layer.weights)} weight sets.")
        new_weights = dict()
        for weights in layer.weights:
            if self._curvature.layer_map.is_bias(weights.name):
                new_weights[weights.name] = tf.add(weights, tf.squeeze(x_b))
            else:
                x_k = tf.reshape(x_k, weights.shape)
                new_weights[weights.name] = tf.add(weights, x_k)

        layer.set_weights(list(new_weights.values()))


class DiagFisherSampler(Sampler):

    def sample_and_replace_weights(self):
        for layer in self._curvature.model.layers:
            if self._curvature.layer_map.is_curvature_eligible(layer.name):
                layer_id = layer.name.split('/')[0]
                var = self._curvature.get_inverse(layer_id)
                x = tf.random.normal(shape=var.shape) * var
                self._split_kernel_bias_weights_and_replace(layer, x)


class BlockDiagFisherSampler(Sampler):

    def sample_and_replace_weights(self):
        for layer in self._curvature.model.layers:
            if self._curvature.layer_map.is_curvature_eligible(layer.name):
                layer_id = layer.name.split('/')[0]
                layer_shape = self._curvature.layer_map.get_layer_shape(layer.name)
                var = self._curvature.get_inverse(layer_id)
                x = tf.linalg.matvec(var, tf.random.normal(shape=[var.shape[0]]))
                x = tf.reshape(x, layer_shape)
                self._split_kernel_bias_weights_and_replace(layer, x)


class KFACSampler(Sampler):

    def sample_and_replace_weights(self):
        for layer in self._curvature.model.layers:
            if self._curvature.layer_map.is_curvature_eligible(layer.name):
                layer_id = layer.name.split('/')[0]
                a, b = self._curvature.get_inverse(layer_id)
                z = tf.random.normal(shape=[a.shape[0], b.shape[0]])
                x = tf.matmul(tf.matmul(a, z), b, transpose_b=True)
                self._split_kernel_bias_weights_and_replace(layer, x)
