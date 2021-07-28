import unittest
import numpy as np
import numpy.testing as npt
from laplace import hooks
from tests.testutils.tensorflow import ModelMocker, RealTfModel


class HooksTest(unittest.TestCase):

    def test_make_hookable(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]

        # when
        hooks.make_hookable(model)

        # then
        self.assertTrue(hasattr(layer1, '_hooks'))
        self.assertTrue(hasattr(layer2, '_hooks'))
        self.assertTrue(hasattr(layer1, 'register_hook'))
        self.assertTrue(hasattr(layer1, 'register_hook'))

    def test_register_hook(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        hooks.make_hookable(model)
        layer_inputs = {}

        def save_inputs(layer, inputs, pre_activations, outputs):
            layer_inputs[layer.name] = inputs

        # when
        layer1.register_hook(layer1, save_inputs)

        # then
        self.assertEqual(1, len(layer1._hooks))
        self.assertTrue(layer1._hooks[0]['enabled'])

    def test_disable_hooks(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        hooks.make_hookable(model)
        layer_inputs = {}

        def save_inputs(layer, inputs, pre_activations, outputs):
            layer_inputs[layer.name] = inputs

        layer1.register_hook(layer1, save_inputs)
        layer2.register_hook(layer2, save_inputs)

        # when
        hooks.disable_hooks(model)

        # then
        self.assertFalse(layer1._hooks[0]['enabled'])
        self.assertFalse(layer2._hooks[0]['enabled'])

    def test_enable_hooks(self):
        # given
        model = ModelMocker.mock_model()
        layer1 = ModelMocker.mock_layer('dense', (2, 3))
        layer2 = ModelMocker.mock_layer('dense_1', (3, 2))
        model.layers = [layer1, layer2]
        hooks.make_hookable(model)
        layer_inputs = {}

        def save_inputs(layer, inputs, pre_activations, outputs):
            layer_inputs[layer.name] = inputs

        layer1.register_hook(layer1, save_inputs)
        layer2.register_hook(layer2, save_inputs)
        hooks.disable_hooks(model)

        # when
        hooks.enable_hooks(model)

        # then
        self.assertTrue(layer1._hooks[0]['enabled'])
        self.assertTrue(layer2._hooks[0]['enabled'])


if __name__ == '__main__':
    unittest.main()