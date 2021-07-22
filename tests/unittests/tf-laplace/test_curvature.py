import unittest
from unittest.mock import patch
from unittest import mock
import tensorflow as tf
import numpy.testing as npt
from laplace.curvature import LayerMap, DiagFisher


@unittest.skip('dev')
class LayerMapTest(unittest.TestCase):

    @patch('tensorflow.keras.layers.Dense', autospec=True)
    @patch('tensorflow.keras.models.Model', autospec=True)
    def setUp(self, model, layer) -> None:
        layer.name = 'dense'

        kernel_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = 'dense/kernel:0'
        kernel_weights.shape = [1, 1]

        bias_weights = mock.create_autospec(tf.Variable)
        bias_weights.name = 'dense/bias:0'
        bias_weights.shape = [1]

        layer.weights = [kernel_weights, bias_weights]

        model.layers = [layer]
        self.dense_model = model

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

    def test_checks_for_bias(self):
        # given
        model = self.dense_model
        # when
        layer_map = LayerMap(model)
        # then
        has_bias = layer_map.has_bias('dense')
        self.assertTrue(has_bias)

    def test_it_exposes_bias_weights(self):
        # given
        model = self.dense_model
        # when
        layer_map = LayerMap(model)
        # then
        weights = layer_map.get_bias_weights('dense')
        self.assertIn('bias', weights['name'])

    def test_it_exposes_kernel_weights(self):
        # given
        model = self.dense_model
        # when
        layer_map = LayerMap(model)
        # then
        weights = layer_map.get_kernel_weights('dense')
        self.assertIn('kernel', weights['name'])

    def test_it_exposes_layer_weights(self):
        # given
        model = self.dense_model
        # when
        layer_map = LayerMap(model)
        # then
        weights = layer_map.get_layer_weights('dense')
        self.assertIn('kernel', weights)
        self.assertIn('bias', weights)

    def test_it_exposes_layer_shape(self):
        # given
        model = self.dense_model
        # when
        layer_map = LayerMap(model)
        # then
        shape = layer_map.get_layer_shape('dense')
        self.assertEqual([2, 1], shape)


class RealTfModel:

    def __init__(self, model):
        self.model = model
        self.input = tf.ones([1,2]) * 1
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


class DiagFisherTest(unittest.TestCase):

    def mock_dens_layer(self, name, shape):
        layer = mock.create_autospec(tf.keras.layers.Dense)
        layer.name = name

        kernel_weights = mock.create_autospec(tf.Variable)
        kernel_weights.name = 'dense/kernel:0'
        kernel_weights.shape = shape

        bias_weights = mock.create_autospec(tf.Variable)
        bias_weights.name = 'dense/bias:0'
        bias_weights.shape = [shape[1]]

        layer.weights = [kernel_weights, bias_weights]

        return layer

    @patch('tensorflow.keras.models.Model', autospec=True)
    def test_update(self, model):
        # given
        layer1 = self.mock_dens_layer('dense', (2, 3))
        layer2 = self.mock_dens_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        gradients = [
            [[2., 2., 2.], [2., 2., 2.]],
            [2., 2., 2.],
            [[3., 3.], [3., 3.], [3., 3.]],
            [1., 1.]
        ]
        diagfisher = DiagFisher(model)

        # when
        diagfisher.update(gradients, 1)

        # then
        stats = {
            'dense': [[4., 4., 4.], [4., 4., 4.], [4., 4., 4.]],
            'dense_1': [[9., 9.], [9., 9.], [9., 9.], [1., 1.]]
        }
        self.assertEqual(len(diagfisher.state), 2)
        for lname, litem in diagfisher.state.items():
            npt.assert_allclose(litem.numpy(), stats[lname])


if __name__ == '__main__':
    unittest.main()
