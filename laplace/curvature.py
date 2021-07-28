import json
import time
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Sequential

from laplace.hooks import make_hookable, disable_hooks


class LayerMap:
    """Helper class to handle kernel and bias weights.

    This class provides methods to handle kernel and bias weights.
    This information is used by the Curvature classes to combine
    kernel and bias weights. It also defines, which layer types
    will be considered for the curvature approximation.

    The information provided for the kernel and bias weights per layer are:
        id: internal identifier
        name: string identifier of the layer or weights
        shape: shape of the respective tensor

    """
    _ELIGIBLE_LAYER_TYPES = ['Dense', 'Conv2D']

    def __init__(self, model: Sequential):
        self._model = model
        self._map = {}

        i = 0
        for layer in model.layers:
            if layer.__class__.__name__ in self._ELIGIBLE_LAYER_TYPES:
                self._map[layer.name] = {'type': layer.__class__.__name__, 'weights': dict()}
                self._map[layer.name]['weights']['kernel'] = {
                    'id': i,
                    'name': layer.weights[0].name,
                    'shape': layer.weights[0].shape
                }
                i += 1
                if len(layer.weights) is 2:
                    self._map[layer.name]['weights']['bias'] = {
                        'id': i,
                        'name': layer.weights[1].name,
                        'shape': layer.weights[1].shape
                    }
                    i += 1

    def is_curvature_eligible(self, layer_id: str) -> bool:
        """Returns whether or not a layer is considered for the curvature approximation.

        Args:
            layer_id: Identifier of the layer

        Returns:
            bool
        """
        layer_id = layer_id.split('/')[0]
        return layer_id in self._map

    def is_bias(self, name: str) -> bool:
        """Determines whether or not given name identifies the bias part of a layer.

        Args:
            name: Identifier of the layer/weights

        Returns:
            bool
        """
        return 'bias' in name

    def has_bias(self, layer_id: str) -> bool:
        """Determines whether or not a layer contains bias weights

        Args:
            layer_id: Identifier of the layer/weights

        Returns:
            bool
        """
        layer_id = layer_id.split('/')[0]
        return 'bias' in self._map[layer_id]['weights']

    def get_bias_weights(self, layer_id: str) -> Dict[str, any]:
        """Returns id, name and shape of the bias weights

        Args:
            layer_id: Identifier of the layer/weights

        Returns:
            Dict[str, any]
        """
        layer_id = layer_id.split('/')[0]
        return self._map[layer_id]['weights']['bias']

    def get_kernel_weights(self, layer_id: str) -> Dict[str, any]:
        """Returns id, name and shape the kernel weights

        Args:
            layer_id: Identifier of the layer/weights

        Returns:
            Dict[str, any]
        """
        layer_id = layer_id.split('/')[0]
        return self._map[layer_id]['weights']['kernel']

    def get_layer_weights(self, layer_id: str) -> Dict[int, Dict[str, any]]:
        """ Returns id, name and shape for the kernel and bias weights of the given layer.

        Args:
            layer_id: Identifier of the layer

        Returns:
            Dict[int, Dict[str, any]]
        """
        return self._map[layer_id]['weights']

    def get_layer_shape(self, layer_id: str) -> List[int]:
        """Returns the shape of the given layer bases on the kept kernel and bias shapes.

        Args:
            layer_id: Identifier of the layer

        Returns:
            List[int]:
        """
        kernel_weights = self.get_kernel_weights(layer_id)
        shape = list(kernel_weights['shape'])
        shape = [np.prod(shape[0:-1])+1, shape[-1]]

        return shape


class Curvature(ABC):
    """Base class for the fisher information matrix approximations.

    Attributes:
        model: The trained model.
        state: The per-layer curvature values
        layer_map: The layer map for the given model
    """

    def __init__(self, model: Sequential):
        self.model = model
        self.state = dict()
        self.layer_map = LayerMap(model)
        self._inverse = dict()

    @abstractmethod
    def save(self, path: str):
        """Abstract method for storing a curvature object on disk at the given path.

        Args:
            path: The path to store the serialised object.

        Returns:
            void
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, model: tf.keras.models.Sequential) -> 'Curvature':
        """Loads a curvature object from disk.

        Args:
            path: The path to load the object from.
            model: The model the curvature relates to.

        Returns:
            Curvature
        """
        pass

    @abstractmethod
    def update(self, gradients: List[tf.Tensor], batch_size: int):
        """Abstract method for updating the curvature matrix with the given gradients.

        Args:
            gradients: The gradients.
            batch_size: The batch size used to compute the gradients.

        Returns:
            void
        """
        pass

    @abstractmethod
    def invert(self, norm: float, scale: float) -> Dict[str, tf.Tensor]:
        """ Abstract method for inverting the curvature matrix.

        Args:
            norm: The hyperparameter :math:`\tau`, which is precision of the prior.
            scale: The hyperparameter :math:`N`.

        Returns:
            Dict[str, tf.Tensor]
        """
        pass

    @classmethod
    def compute(cls,
                model: keras.Sequential,
                dataset: tf.data.Dataset,
                criterion: keras.losses.Loss) -> 'Curvature':
        """Factory method to compute an approximation to the curvature.

        Code inspired by: https://github.com/DLR-RM/curvature

        Args:
            model: The trained model
            dataset: The training dataset
            criterion: The loss function

        Returns:
            Curvature
        """
        curvature = cls(model)
        total_batches = len(list(dataset))
        batch = 1
        for x, y in dataset:
            for i in range(10):
                with tf.GradientTape() as tape:
                    logits = model(x, training=False)
                    dist = tfp.distributions.Bernoulli(logits=logits)
                    labels = dist.sample()
                    loss_val = criterion(labels, logits)
                grads = tape.gradient(loss_val, model.trainable_weights)
                curvature.update(grads, batch_size=x.shape[0])
            batch += 1
            if total_batches > 50 and batch % int(total_batches/50) == 0:
                print(".", end='', flush=True)
        return curvature

    def get_inverse(self, layer_id: str = None):
        """
        Returns the inverse curvature matrix.

        Args:
            layer_id: The identifier of the network layer. If set, only the
                inverse associated with the specified layer will be returned.

        Returns:
            Dict[str, tf.Tensor]

        Raises:
            RuntimeError: if the inverse was not computed before.
        """
        if len(self._inverse) == 0:
            raise RuntimeError("Inverse is not available. Call Curvature.invert(norm, scale) first.")
        if layer_id is not None:
            return self._inverse[layer_id]
        else:
            return self._inverse

    def _combine_kernel_bias_gradients_for_layer(self,
                                                 gradients: List[tf.Tensor],
                                                 layer: keras.layers.Layer) -> tf.Tensor:
        """
        Combines the separate gradient tensors for kernel and bias into one tensor.

        Args:
            gradients: All gradient tensors for all layers, in the same order as in in model.weights
            layer: The layer to combine the gradients for.

        Returns:
            One tf.Tensor, containing the gradients for kernel and bias weights
        """
        kernel_weights = self.layer_map.get_kernel_weights(layer.name)
        bias_weights = self.layer_map.get_bias_weights(layer.name)
        kernel_gradients = itemgetter(kernel_weights['id'])(gradients)
        kernel_gradients = tf.reshape(kernel_gradients, [-1, kernel_weights['shape'][-1]])
        result = tf.concat([kernel_gradients, tf.expand_dims(gradients[bias_weights['id']], axis=0)], axis=0)
        return result


class DiagFisher(Curvature):
    """The diagonal fisher information matrix approximation.

    The diagonal fisher is defined as :math:`F_{diag} = \mathrm{diag}(F)`
    with F being the full Fisher Information Matrix (see FullFisher)

    Code inspired by: https://github.com/DLR-RM/curvature
    """

    def save(self, path: str):
        raise NotImplementedError('Method not implemented yet.')
        pass

    @classmethod
    def load(cls, path: str, model: tf.keras.models.Sequential) -> 'Curvature':
        raise NotImplementedError('Method not implemented yet.')
        pass

    def update(self, gradients: List[tf.Tensor], batch_size: int):
        """Updates the diagonal fisher information matrix by given gradients.

        Args:
            gradients: The gradients.
            batch_size: The batch size used to compute the gradients.

        Returns:
            void
        """
        for layer in self.model.layers:
            if self.layer_map.is_curvature_eligible(layer.name):
                g = self._combine_kernel_bias_gradients_for_layer(gradients, layer)
                g = g ** 2 * batch_size
                if layer.name in self.state:
                    self.state[layer.name] += g
                else:
                    self.state[layer.name] = g

    def invert(self, norm: float, scale: float) -> Dict[str, tf.Tensor]:
        """Invert the diagonal fisher matrix.

        The inverse matrix is cached as a member. If `invert` is called a second time,
        the previous version will be replaced.

        Args:
            norm: The hyperparameter :math:`\tau`. :math:`\tau I` is added to each diagonal element.
            scale: The hyperparameter :math:`N`, which is used to scale each diagonal element.

        Returns:
            Dict[str, tf.Tensor]
        """
        self._inverse.clear()
        for layer_id in self.state:
            inv = tf.math.reciprocal(scale * self.state[layer_id] + norm)
            self._inverse[layer_id] = tf.math.sqrt(inv)
        return self._inverse


class BlockDiagFisher(Curvature):
    """The block diagonal fisher information matrix.

    The full Fisher Information Matrix (FIM) can be defined as the outer product
    of the gradient of the network's loss E w.r.t. its weights W
    :math:`F = \mathbb{E}\left[\nabla_W E(W) \nabla_W E(W)^T\right]`

    The block diagonal fisher information matrix contains the layer-wise diagonal blocks of the full FIM.

    Code inspired by: https://github.com/DLR-RM/curvature

    """

    def save(self, path: str):
        raise NotImplementedError('Method not implemented yet.')
        pass

    @classmethod
    def load(cls, path: str, model: tf.keras.models.Sequential) -> 'Curvature':
        raise NotImplementedError('Method not implemented yet.')
        pass

    def update(self, gradients: List[tf.Tensor], batch_size: int):
        """Updates the block-diagonal FIM with the given gradients.

        Args:
            gradients: The gradients.
            batch_size: The batch size used to compute the gradients.

        Returns:
            void
        """
        for layer in self.model.layers:
            if self.layer_map.is_curvature_eligible(layer.name):
                g = self._combine_kernel_bias_gradients_for_layer(gradients, layer)
                g = tf.reshape(g, [-1])
                g = tf.tensordot(g, g, axes=0) * batch_size
                if layer.name in self.state:
                    self.state[layer.name] += g
                else:
                    self.state[layer.name] = g

    def invert(self, norm: float, scale: float) -> Dict[str, tf.Tensor]:
        """Invert the block-diagonal fisher matrix.

        The inverse matrix is cached as a member. If `invert` is called a second time,
        the previous version will be replaced.

        Args:
            norm: The hyperparameter :math:`\tau`. :math:`\tau I` is added to the diagonal
                elements of each block.
            scale: The hyperparameter :math:`N`, which is used to scale each block.

        Returns:
            Dict[str, tf.Tensor]

        """
        self._inverse.clear()
        for layer_id in self.state:
            reg = tf.linalg.diag(tf.fill([self.state[layer_id].shape[0]], norm))
            self._inverse[layer_id] = tf.linalg.inv(tf.add(scale * self.state[layer_id], reg))
        return self._inverse


class KFAC(Curvature):
    """The Kronecker factored approximation of the fisher information matrix.

    For a single data point, each diagonal block :math:`\lambda` of the Fisher
    Information Matrix can be expressed as a Kronecker product of two factores

    .. math::
        H_\lambda = \mathcal{Q}_\lambda \otimes \mathcal{H}_\lambda

    * :math:`\mathcal{Q}_\lambda` denotes
      the covariances of the incoming activations from the previous layer
      :math:`\lambda-1`

      .. math::
        \mathcal{Q}_\lambda = a_{\lambda-1} a_{\lambda-1}^T

    * :math:`\mathcal{H}_\lambda` denotes the Hessian of the loss function
      w.r.t. the linear pre-activations in layer :math:`\lambda`

      .. math::
        \mathcal{H}_\lambda = \partial^2 / \partial h_\lambda \partial h_\lambda

    """

    def __init__(self,
                 model: Sequential):
        """ Constructor of the KFAC approximation

        The constructor mokey-patches the model object to assign hooks
        to the layers. These hooks are used to grab
            * the pre-activations
            * the inputs
        of each layer.

        Args:
            model: The model to approximate the curvature for.
        """
        self._layer_inputs = dict()
        self._layer_preactivations = dict()
        self._layer_outputs = dict()
        self._tape = None
        make_hookable(model)
        super().__init__(model)
        for layer in model.layers:
            if self.layer_map.is_curvature_eligible(layer.name):
                self._layer_inputs[layer.name] = None
                self._layer_preactivations[layer.name] = None
                layer.register_hook(layer, self._save_inputs)
                layer.register_hook(layer, self._watch_preactivations)

    def _save_inputs(self,
                     layer: tf.keras.layers.Layer,
                     inputs: tf.Tensor,
                     pre_activation: tf.Tensor,
                     output: tf.Tensor):
        self._layer_inputs[layer.name] = inputs

    def _watch_preactivations(self,
                              layer: tf.keras.layers.Layer,
                              inputs: tf.Tensor,
                              pre_activation: tf.Tensor,
                              output: tf.Tensor):
        self._layer_preactivations[layer.name] = pre_activation
        self._tape.watch(self._layer_preactivations[layer.name])

    def set_tape(self,
                 tape: tf.GradientTape):
        """Sets the gradient tape.

        Sets the gradient tape that is used to calculate the gradients
        wrt to the linear preactivations.

        Args:
            tape: The gradient tape

        Returns:
            void
        """
        self._tape = tape

    def get_layer_preactivations(self) -> Dict:
        """ Returns all linear pre-activations.

        Returns:
            Dict
        """
        return self._layer_preactivations

    def save(self,
             path: str):
        """Stores a KFAC object on disk.

        The object is stored in a json format.

        Args:
            path: The file-system path to write the file to

        Returns:
            void
        """
        json_data = {'type': self.__class__.__name__, 'layers': {}}
        for layer, factors in self.state.items():
            json_data['layers'][layer] = []
            for factor in factors:
                json_data['layers'][layer].append({'shape': list(factor.shape), 'data': factor.numpy().tolist(), 'dtype': factor.dtype.name})

        with open(path, 'w') as outfile:
            json.dump(json_data, outfile)

    @classmethod
    def load(cls,
             path: str,
             model: keras.models.Sequential) -> 'Curvature':
        """Loads a KFAC object from disk stored in json format.

        Args:
            path: The filesystem path.
            model: The model the loaded object relates to.

        Returns:
            KFAC
        """
        with open(path) as json_file:
            data = json.load(json_file)
            curvature = cls(model)
            for l_name, l_data in data['layers'].items():
                curvature.state[l_name] = []
                for factor in l_data:
                    t = tf.convert_to_tensor(factor['data'], dtype=tf.as_dtype(factor['dtype']))
                    curvature.state[l_name].append(t)

        disable_hooks(model)
        return curvature

    @classmethod
    def compute(cls,
                model: keras.Sequential,
                dataset: tf.data.Dataset,
                criterion: keras.losses.Loss) -> 'Curvature':
        """Factory method to compute the Kronecker factored approximation to the FIM

        Args:
            model: The model to approximate the curvature for.
            dataset: The training dataset given in batches.
            criterion: The loss function.

        Returns:
            KFAC
        """
        kfac = cls(model)
        for x, y in dataset:
            with tf.GradientTape() as tape:
                kfac.set_tape(tape)
                logits = model(x, training=False)
                dist = tfp.distributions.Bernoulli(logits=logits)
                labels = dist.sample()
                loss_val = criterion(labels, logits)
            grads_pa = tape.gradient(loss_val, kfac.get_layer_preactivations())
            kfac.update(grads_pa, batch_size=x.shape[0])
        disable_hooks(model)
        return kfac

    def update(self, gradients: List[tf.Tensor], batch_size: int):
        """Update the Kronecker Factors.

        Args:
            gradients: The gradients, taken with respect to pre-activations of each layer.
            batch_size: The batch size used to compute the gradients.

        Returns:
            void
        """
        for index, layer in enumerate(self.model.layers):
            if self.layer_map.is_curvature_eligible(layer.name):
                # factor Q
                # covariance of incoming activations (layerwise): Q_l = a_{l-1} a_{l-1}^T

                if layer.__class__.__name__ is 'Conv2D':
                    a = self._layer_inputs[layer.name]
                    a = tf.image.extract_patches(images=a,
                                                 sizes=[1] + list(layer.kernel_size) + [1],
                                                 strides=[1] + list(layer.strides) + [1],
                                                 padding=layer.padding.upper(),
                                                 rates=[1] + list(layer.dilation_rate) + [1])
                    a = tf.reshape(a, [a.shape[-1], -1])
                else:
                    a = tf.transpose(self._layer_inputs[layer.name])
                if self.layer_map.has_bias(layer.name):
                    ones = tf.ones(a[:1].shape)
                    a = tf.concat([a, ones], axis=0)
                factor_Q = tf.matmul(a, a, transpose_b=True) / float(a.shape[1])

                # factor H
                # pre-activation hessian
                dh = gradients[layer.name] * gradients[layer.name].shape[0]

                if layer.__class__.__name__ is 'Conv2D':
                    dh = tf.reshape(dh, [dh.shape[-1], -1])
                else:
                    dh = tf.transpose(dh)
                factor_H = tf.matmul(dh, dh, transpose_b=True) / float(dh.shape[1])

                # Expectation
                if layer.name in self.state:
                    self.state[layer.name][0] += factor_Q
                    self.state[layer.name][1] += factor_H
                else:
                    self.state[layer.name] = [factor_Q, factor_H]

    def invert(self, norm: float, scale: float) -> Dict[str, tf.Tensor]:
        """Inverts the Kronecker factored approximation of the FIM

        The inverse matrix is cached as a member. If `invert` is called a second time,
        the previous version will be replaced.

        Args:
            norm: The hyperparameter :math:`\tau`. :math:`\tau I` is added to the diagonal
                elements of each block.
            scale: The hyperparameter :math:`N`, which is used to scale each block.

        Returns:
            Dict[str, tf.Tensor]

        """
        self._inverse.clear()
        for layer_id in self.state:
            factor_Q, factor_H = self.state[layer_id]

            diag_tau_Q = tf.linalg.diag(tf.fill([factor_Q.shape[0]], norm ** 0.5))
            diag_tau_H = tf.linalg.diag(tf.fill([factor_H.shape[0]], norm ** 0.5))

            reg_Q = scale ** 0.5 * factor_Q + diag_tau_Q
            reg_H = scale ** 0.5 * factor_H + diag_tau_H

            reg_Q = (reg_Q + tf.transpose(reg_Q)) / 2.0
            reg_H = (reg_H + tf.transpose(reg_H)) / 2.0

            try:
                inv_Q = tf.linalg.inv(reg_Q)
                inv_H = tf.linalg.inv(reg_H)
                cholesky_Q = tf.linalg.cholesky(inv_Q)
                cholesky_H = tf.linalg.cholesky(inv_H)
            except tf.errors.InvalidArgumentError as e:
                print("Cholesky decomposition with Tensorflow failed. Switching to Numpy.")
                inv_Q = np.linalg.inv(reg_Q.numpy())
                inv_H = np.linalg.inv(reg_H.numpy())
                cholesky_Q = tf.convert_to_tensor(np.linalg.cholesky(inv_Q))
                cholesky_H = tf.convert_to_tensor(np.linalg.cholesky(inv_H))

            self._inverse[layer_id] = (cholesky_Q, cholesky_H)

        return self._inverse
