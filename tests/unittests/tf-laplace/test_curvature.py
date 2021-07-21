import unittest
from unittest.mock import patch
from laplace.curvature import LayerMap
import tensorflow as tf


class LayerMapTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dense_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, input_dim=1, activation='linear', name='dense')
        ])

        self.conv_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", activation='linear',
                                   input_shape=(1, 1, 1), name='conv'),
        ])

        self.pooling_model = tf.keras.models.Sequential([
            tf.keras.layers.MaxPooling2D(pool_size=(1, 1), name='pooling'),
        ])

    def test_considers_dense_layers(self):
        # given
        model = self.dense_model
        layer_map = LayerMap(model)
        # when
        is_layer_considered = layer_map.is_curvature_eligible('dense')
        # then
        self.assertTrue(is_layer_considered)

    def test_considers_conv_layers(self):
        # given
        model = self.conv_model
        layer_map = LayerMap(model)
        # when
        is_layer_considered = layer_map.is_curvature_eligible('conv')
        # then
        self.assertTrue(is_layer_considered)

    def test_does_not_consider_pooling_layers(self):
        # given
        model = self.pooling_model
        layer_map = LayerMap(model)
        # when
        is_layer_considered = layer_map.is_curvature_eligible('pooling')
        # then
        self.assertFalse(is_layer_considered)

    def test_checks_for_bias(self):
        # given
        model = self.dense_model
        layer_map = LayerMap(model)
        # when
        has_bias = layer_map.has_bias('dense')
        # then
        self.assertTrue(has_bias)

    def test_it_exposes_bias_weights(self):
        # given
        model = self.dense_model
        layer_map = LayerMap(model)
        # when
        weights = layer_map.get_bias_weights('dense')
        # then
        self.assertIn('bias', weights['name'])

    def test_it_exposes_kernel_weights(self):
        # given
        model = self.dense_model
        layer_map = LayerMap(model)
        # when
        weights = layer_map.get_kernel_weights('dense')
        # then
        self.assertIn('kernel', weights['name'])

    def test_it_exposes_layer_weights(self):
        # given
        model = self.dense_model
        layer_map = LayerMap(model)
        # when
        weights = layer_map.get_layer_weights('dense')
        # then
        self.assertIn('kernel', weights)
        self.assertIn('bias', weights)

    def test_it_exposes_layer_shape(self):
        # given
        model = self.dense_model
        layer_map = LayerMap(model)
        # when
        shape = layer_map.get_layer_shape('dense')
        # then
        self.assertEqual([2, 1], shape)




if __name__ == '__main__':
    unittest.main()
