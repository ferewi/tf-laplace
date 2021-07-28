import unittest
from unittest.mock import patch
import numpy as np

from laplace.curvature import DiagFisher, BlockDiagFisher, KFAC
from laplace.sampler import Sampler, DiagFisherSampler, BlockDiagFisherSampler, KFACSampler
from tests.testutils.tensorflow import ModelMocker


class SamplerTest(unittest.TestCase):

    def test_it_creates_the_sampler_according_to_the_curvature(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]

        diagfisher = DiagFisher(model)
        blockfisher = BlockDiagFisher(model)
        kfac = KFAC(model)

        # when
        diagsampler = Sampler.create(diagfisher, tau=1., n=1.)
        blocksampler = Sampler.create(blockfisher, tau=1., n=1.)
        kfacsampler = Sampler.create(kfac, tau=1., n=1.)

        # then
        self.assertIsInstance(diagsampler, DiagFisherSampler)
        self.assertIsInstance(blocksampler, BlockDiagFisherSampler)
        self.assertIsInstance(kfacsampler, KFACSampler)


class DiagFisherSamplerTest(unittest.TestCase):

    @patch('tensorflow.add')
    def test_sample_and_replace_weights(self, add_func):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        add_func.side_effect = lambda x, y: y

        diagfisher = DiagFisher(model)
        diagfisher.state = {
            'dense': np.array([[4., 4., 4.], [4., 4., 4.], [4., 4., 4.]], dtype=np.float32),
            'dense_1': np.array([[9., 9.], [9., 9.], [9., 9.], [1., 1.]], dtype=np.float32)
        }
        sampler = DiagFisherSampler(diagfisher, tau=1., n=1.)

        # assert new weights have right shape
        def assert_weight_layer_1(x):
            self.assertEqual([2, 3], x[0].shape)
            self.assertEqual([3, ], x[1].shape)

        def assert_weight_layer_2(x):
            self.assertEqual([3, 2], x[0].shape)
            self.assertEqual([2, ], x[1].shape)
        layer1.set_weights = assert_weight_layer_1
        layer2.set_weights = assert_weight_layer_2

        # when
        sampler.sample_and_replace_weights()


class BlockFisherSamplerTest(unittest.TestCase):

    @patch('tensorflow.add')
    def test_sample_and_replace_weights(self, add_func):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        add_func.side_effect = lambda x, y: y

        blockfisher = BlockDiagFisher(model)
        blockfisher.state = {
            'dense': np.ones([9, 9]) * 4,
            'dense_1': np.repeat([[9., 9., 9., 9., 9., 9., 3., 3.]], 8, axis=0)
        }
        blockfisher.state['dense_1'][6] = blockfisher.state['dense_1'][6] / 3
        blockfisher.state['dense_1'][7] = blockfisher.state['dense_1'][7] / 3
        sampler = BlockDiagFisherSampler(blockfisher, tau=1., n=1.)

        # assert new weights have right shape
        def assert_weight_layer_1(x):
            self.assertEqual([2, 3], x[0].shape)
            self.assertEqual([3, ], x[1].shape)

        def assert_weight_layer_2(x):
            self.assertEqual([3, 2], x[0].shape)
            self.assertEqual([2, ], x[1].shape)
        layer1.set_weights = assert_weight_layer_1
        layer2.set_weights = assert_weight_layer_2

        # when
        sampler.sample_and_replace_weights()


class KFACSamplerTest(unittest.TestCase):

    @patch('tensorflow.add')
    def test_sample_and_replace_weights(self, add_func):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        add_func.side_effect = lambda x, y: y

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
        sampler = KFACSampler(kfac, tau=1., n=1.)

        # assert new weights have right shape
        def assert_weight_layer_1(x):
            self.assertEqual([2, 3], x[0].shape)
            self.assertEqual([3, ], x[1].shape)

        def assert_weight_layer_2(x):
            self.assertEqual([3, 2], x[0].shape)
            self.assertEqual([2, ], x[1].shape)
        layer1.set_weights = assert_weight_layer_1
        layer2.set_weights = assert_weight_layer_2

        # when
        sampler.sample_and_replace_weights()


if __name__ == '__main__':
    unittest.main()