import unittest
from unittest.mock import patch
from unittest import mock

from laplace.curvature import LayerMap
import tensorflow as tf


class LayerMapTest(unittest.TestCase):

    @patch('tensorflow.keras.layers.Dense', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_considers_dense_layers(self, model, layer):
        # given
        layer.name = 'dense'
        model.layers = [layer]
        # when
        layer_map = LayerMap(model)
        # then
        is_layer_considered = layer_map.is_curvature_eligible('dense')
        self.assertTrue(is_layer_considered)

    @patch('tensorflow.keras.layers.Conv2D', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_considers_conv_layers(self, model, layer):
        # given
        layer.name = 'conv'
        model.layers = [layer]
        # when
        layer_map = LayerMap(model)
        # then
        is_layer_considered = layer_map.is_curvature_eligible('conv')
        self.assertTrue(is_layer_considered)

    @patch('tensorflow.keras.layers.MaxPooling2D', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_does_not_consider_pooling_layers(self, model, layer):
        # given
        layer.name = 'pooling'
        model.layers = [layer]
        # when
        layer_map = LayerMap(model)
        # then
        is_layer_considered = layer_map.is_curvature_eligible('pooling')
        self.assertFalse(is_layer_considered)

    @patch('tensorflow.keras.layers.Dense', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_checks_for_bias(self, model, layer):
        # given
        layer.name = 'dense'
        kernel_weights = mock.create_autospec(tf.Variable)
        bias_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = 'dense/kernel:0'
        bias_weights.name = 'dense/bias:0'
        layer.weights = [kernel_weights, bias_weights]
        model.layers = [layer]
        # when
        layer_map = LayerMap(model)
        # then
        has_bias = layer_map.has_bias('dense')
        self.assertTrue(has_bias)

    @patch('tensorflow.keras.layers.Dense', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_it_exposes_bias_weights(self, model, layer):
        # given
        layer.name = 'dense'
        kernel_weights = mock.create_autospec(tf.Variable)
        bias_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = 'dense/kernel:0'
        bias_weights.name = 'dense/bias:0'
        layer.weights = [kernel_weights, bias_weights]
        model.layers = [layer]
        # when
        layer_map = LayerMap(model)
        # then
        weights = layer_map.get_bias_weights('dense')
        self.assertIn('bias', weights['name'])

    @patch('tensorflow.keras.layers.Dense', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_it_exposes_kernel_weights(self, model, layer):
        # given
        layer.name = 'dense'
        kernel_weights = mock.create_autospec(tf.Variable)
        bias_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = 'dense/kernel:0'
        bias_weights.name = 'dense/bias:0'
        layer.weights = [kernel_weights, bias_weights]
        model.layers = [layer]

        # when
        layer_map = LayerMap(model)
        weights = layer_map.get_kernel_weights('dense')
        # then
        self.assertIn('kernel', weights['name'])

    @patch('tensorflow.keras.layers.Dense', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_it_exposes_layer_weights(self, model, layer):
        # given
        layer.name = 'dense'
        kernel_weights = mock.create_autospec(tf.Variable)
        bias_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = 'dense/kernel:0'
        bias_weights.name = 'dense/bias:0'
        layer.weights = [kernel_weights, bias_weights]
        model.layers = [layer]
        # when
        layer_map = LayerMap(model)
        weights = layer_map.get_layer_weights('dense')
        # then
        self.assertIn('kernel', weights)
        self.assertIn('bias', weights)

    @patch('tensorflow.keras.layers.Dense', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_it_exposes_layer_shape(self, model, layer):
        # given
        layer.name = 'dense'
        kernel_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = 'dense/kernel:0'
        kernel_weights.shape = [1, 1]
        layer.weights = [kernel_weights]
        model.layers = [layer]
        # when
        layer_map = LayerMap(model)
        # then
        shape = layer_map.get_layer_shape('dense')
        self.assertEqual([2, 1], shape)


if __name__ == '__main__':
    unittest.main()
