import unittest
from unittest.mock import patch
from unittest import mock
import tensorflow as tf
import numpy as np
import numpy.testing as npt
from laplace.curvature import LayerMap, DiagFisher, BlockDiagFisher, KFAC
from tests.testutils.tensorflow import ModelMocker


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


class DiagFisherTest(unittest.TestCase):

    def test_update_first_iteration(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
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
        expected_state = {
            'dense': [[4., 4., 4.], [4., 4., 4.], [4., 4., 4.]],
            'dense_1': [[9., 9.], [9., 9.], [9., 9.], [1., 1.]]
        }
        self.assertEqual(len(diagfisher.state), 2)
        for lname, litem in diagfisher.state.items():
            npt.assert_allclose(litem.numpy(), expected_state[lname])

    def test_update_second_iteration(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        gradients = [
            [[2., 2., 2.], [2., 2., 2.]],
            [2., 2., 2.],
            [[3., 3.], [3., 3.], [3., 3.]],
            [1., 1.]
        ]
        diagfisher = DiagFisher(model)
        diagfisher.state = {
            'dense': [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
            'dense_1': [[1., 1.], [1., 1.], [1., 1.], [1., 1.]]
        }

        # when
        diagfisher.update(gradients, 1)

        # then
        expected_state = {
            'dense': np.array([[5., 5., 5.], [5., 5., 5.], [5., 5., 5.]]),
            'dense_1': np.array([[10., 10.], [10., 10.], [10., 10.], [2., 2.]])
        }
        self.assertEqual(len(diagfisher.state), 2)
        for lname, litem in diagfisher.state.items():
            npt.assert_allclose(litem.numpy(), expected_state[lname])

    def test_invert(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        diagfisher = DiagFisher(model)
        diagfisher.state = {
            'dense': np.array([[4., 4., 4.], [4., 4., 4.], [4., 4., 4.]]),
            'dense_1': np.array([[9., 9.], [9., 9.], [9., 9.], [1., 1.]])
        }

        # when
        inverse = diagfisher.invert(tau=1., n=1.)

        # then
        expected_inverse = {
            'dense': np.array([[0.4472136, 0.4472136, 0.4472136],
                               [0.4472136, 0.4472136, 0.4472136],
                               [0.4472136, 0.4472136, 0.4472136]]),
            'dense_1': np.array([[0.31622777, 0.31622777],
                                 [0.31622777, 0.31622777],
                                 [0.31622777, 0.31622777],
                                 [0.70710678, 0.70710678]]),
        }
        self.assertEqual(len(inverse), 2)
        for lname, litem in inverse.items():
            npt.assert_allclose(litem.numpy(), expected_inverse[lname])


class BlockDiagFisherTest(unittest.TestCase):

    def test_update_first_iteration(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        gradients = [
            [[2., 2., 2.], [2., 2., 2.]],
            [2., 2., 2.],
            [[3., 3.], [3., 3.], [3., 3.]],
            [1., 1.]
        ]
        blockfisher = BlockDiagFisher(model)

        # when
        blockfisher.update(gradients, 1)

        # then
        expected_state = {
            'dense': np.ones([9, 9])*4,
            'dense_1': np.repeat([[9., 9., 9., 9., 9., 9., 3., 3.]], 8, axis=0)
        }
        expected_state['dense_1'][6] = expected_state['dense_1'][6]/3
        expected_state['dense_1'][7] = expected_state['dense_1'][7]/3
        self.assertEqual(len(blockfisher.state), 2)
        for lname, litem in blockfisher.state.items():
            npt.assert_allclose(litem.numpy(), expected_state[lname])

    def test_update_second_iteration(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        gradients = [
            [[2., 2., 2.], [2., 2., 2.]],
            [2., 2., 2.],
            [[3., 3.], [3., 3.], [3., 3.]],
            [1., 1.]
        ]
        blockfisher = BlockDiagFisher(model)
        blockfisher.state = {
            'dense': np.ones([9, 9])*1,
            'dense_1': np.repeat([[1., 1., 1., 1., 1., 1., 1., 1.]], 8, axis=0)
        }

        # when
        blockfisher.update(gradients, 1)

        # then
        expected_state = {
            'dense': np.ones([9, 9]) * 5,
            'dense_1': np.repeat([[10., 10., 10., 10., 10., 10., 4., 4.]], 8, axis=0)
        }
        expected_state['dense_1'][6] = [4., 4., 4., 4., 4., 4., 2., 2.]
        expected_state['dense_1'][7] = [4., 4., 4., 4., 4., 4., 2., 2.]
        self.assertEqual(len(blockfisher.state), 2)
        for lname, litem in blockfisher.state.items():
            npt.assert_allclose(litem.numpy(), expected_state[lname])

    def test_invert(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]

        blockfisher = BlockDiagFisher(model)
        blockfisher.state = {
            'dense': np.ones([9, 9]) * 4,
            'dense_1': np.repeat([[9., 9., 9., 9., 9., 9., 3., 3.]], 8, axis=0)
        }
        blockfisher.state['dense_1'][6] = blockfisher.state['dense_1'][6] / 3
        blockfisher.state['dense_1'][7] = blockfisher.state['dense_1'][7] / 3

        # when
        inverse = blockfisher.invert(tau=1., n=1.)

        # then
        expected_inverse = {
            'dense': np.array([
                [ 0.89189196, -0.10810812, -0.10810812, -0.10810812, -0.10810812, -0.10810813, -0.10810812, -0.10810813, -0.10810812],
                [-0.10810812,  0.89189196, -0.10810812, -0.10810812, -0.10810812, -0.10810813, -0.10810812, -0.10810813, -0.10810812],
                [-0.10810812, -0.10810812,  0.89189196, -0.10810811, -0.10810811, -0.10810811, -0.10810811, -0.10810811, -0.10810811],
                [-0.10810812, -0.10810812, -0.10810811,  0.89189196, -0.10810811, -0.10810811, -0.10810811, -0.10810811, -0.10810811],
                [-0.10810812, -0.10810812, -0.10810811, -0.10810811,  0.89189196, -0.10810811, -0.10810811, -0.10810811, -0.10810811],
                [-0.10810813, -0.10810813, -0.10810811, -0.10810811, -0.10810811,  0.89189196, -0.10810811, -0.10810811, -0.10810811],
                [-0.10810812, -0.10810812, -0.10810811, -0.10810811, -0.10810811, -0.10810811,  0.89189196, -0.10810811, -0.10810811],
                [-0.10810813, -0.10810813, -0.10810811, -0.10810811, -0.10810811, -0.10810811, -0.10810811,  0.89189196, -0.10810811],
                [-0.10810812, -0.10810812, -0.10810811, -0.10810811, -0.10810811, -0.10810811, -0.10810811, -0.10810811,  0.8918919]
            ], dtype=np.float32),
            'dense_1': np.array([
                 [0.8421055,   -0.15789483, -0.15789482, -0.1578948,  -0.15789478, -0.15789479, -0.05263155, -0.05263155],
                 [-0.15789483,  0.8421052,  -0.15789473, -0.1578947,  -0.15789469, -0.1578947,  -0.05263159, -0.05263159],
                 [-0.15789482, -0.15789473,  0.84210527, -0.15789473, -0.15789473, -0.15789473, -0.0526316,  -0.0526316],
                 [-0.15789479, -0.1578947,  -0.15789473,  0.84210527, -0.15789473, -0.15789475, -0.0526316,  -0.0526316],
                 [-0.15789478, -0.15789469, -0.15789473, -0.15789473,  0.8421052,  -0.15789473, -0.0526316,  -0.0526316],
                 [-0.15789479, -0.1578947,  -0.15789473, -0.15789475, -0.15789473,  0.84210527, -0.0526316,  -0.05263161],
                 [-0.05263154, -0.05263159, -0.0526316,  -0.0526316,  -0.0526316,  -0.0526316,   0.98245627, -0.01754383],
                 [-0.05263154, -0.05263159, -0.0526316,  -0.0526316,  -0.0526316,  -0.05263161, -0.01754383,  0.9824563]
            ], dtype=np.float32)
        }
        self.assertEqual(len(inverse), 2)
        for lname, litem in inverse.items():
            npt.assert_allclose(litem.numpy(), expected_inverse[lname], rtol=1e-6, atol=1e-7)


class KFACTest(unittest.TestCase):
    def test_update_first_iteration(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]

        kfac = KFAC(model)
        kfac._layer_inputs['dense'] = [[1., 1.]]
        kfac._layer_inputs['dense_1'] = [[3., 3., 3.]]
        grads_preactivations = {
            'dense': np.array([[18., 18., 18.]]),
            'dense_1': np.array([[9., 9.]]),
        }

        # when
        kfac.update(grads_preactivations, 1)

        # then
        expected_state = {
            'dense': [
                np.ones([3, 3]),
                np.ones([3, 3]) * 324
            ],
            'dense_1': [
                np.array([[9., 9., 9., 3.], [9., 9., 9., 3.], [9., 9., 9., 3.], [3., 3., 3., 1.]]),
                np.ones([2, 2]) * 81
            ]
        }
        self.assertEqual(len(kfac.state), 2)
        for lname, litem in kfac.state.items():
            Q, H = litem[0].numpy(), litem[1].numpy()
            npt.assert_allclose(Q, expected_state[lname][0])
            npt.assert_allclose(H, expected_state[lname][1])

    def test_update_second_iteration(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]

        kfac = KFAC(model)
        kfac._layer_inputs['dense'] = [[1., 1.]]
        kfac._layer_inputs['dense_1'] = [[3., 3., 3.]]
        grads_preactivations = {
            'dense': np.array([[18., 18., 18.]]),
            'dense_1': np.array([[9., 9.]]),
        }
        kfac.state = {
            'dense': [np.ones([3, 3]), np.ones([3, 3])],
            'dense_1': [np.ones([4, 4]), np.ones([2, 2])]
        }

        # when
        kfac.update(grads_preactivations, 1)

        # then
        expected_state = {
            'dense': [
                np.ones([3, 3]) * 2,
                np.ones([3, 3]) * 325
            ],
            'dense_1': [
                np.array([[10., 10., 10., 4.], [10., 10., 10., 4.], [10., 10., 10., 4.], [4., 4., 4., 2.]]),
                np.array([[82., 82.], [82., 82.]])
            ]
        }
        self.assertEqual(len(kfac.state), 2)
        for lname, litem in kfac.state.items():
            Q, H = litem[0].numpy(), litem[1].numpy()
            npt.assert_allclose(Q, expected_state[lname][0])
            npt.assert_allclose(H, expected_state[lname][1])

    def test_invert(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]

        kfac = KFAC(model)
        kfac.state = {
            'dense': [
                np.ones([3, 3]),
                np.ones([3, 3]) * 324
            ],
            'dense_1': [
                np.array([[9., 9., 9., 3.], [9., 9., 9., 3.], [9., 9., 9., 3.], [3., 3., 3., 1.]]),
                np.array([[81., 81.], [81., 81.]])
            ]
        }

        # when
        inv = kfac.invert(tau=1., n=1.)

        # then
        expected_inv = {
            'dense': [
                np.array([[0.8660254, 0., 0.],
                 [-0.28867513, 0.8164966, 0.],
                 [-0.2886751, -0.40824828, 0.70710677]]),
                np.array([[0.81670785, 0., 0.],
                 [-0.40772474, 0.7076513, 0.],
                 [-0.4077247, -0.70547396, 0.05546965]])
            ],
            'dense_1': [
                np.array([[0.8304549, 0., 0., 0.],
                 [-0.3737047, 0.7416198, 0., 0.],
                 [-0.37370473, -0.6067798, 0.42640147, 0.],
                 [-0.12456819, -0.20225996, -0.6396021, 0.7071068]]),
                np.array([[0.7092719, 0.],
                 [-0.7006222, 0.11043184]]),
            ]
        }
        self.assertEqual(len(inv), 2)
        for lname, factors in inv.items():
            Q, H = factors[0].numpy(), factors[1].numpy()
            npt.assert_allclose(Q, expected_inv[lname][0], rtol=1e-6, atol=1e-7)
            npt.assert_allclose(H, expected_inv[lname][1], rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
