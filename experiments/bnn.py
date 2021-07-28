import numpy as np
import tensorflow as tf
from laplace.sampler import Sampler

class BNN:

    def __init__(self, model: tf.keras.Sequential, sampler: Sampler):
        self._model = model
        self._sampler = sampler

    def evaluate(self, data: np.ndarray, n_samples: int, apply_sigmoid=True) -> np.ndarray:
        num_datapoints = data.shape[0]
        num_classes = self._model.layers[-1].output.shape[1]
        predictions = np.zeros([n_samples, num_datapoints, num_classes], dtype=np.float32)

        posterior_mean = self._model.get_weights()
        for sample in range(n_samples):
            self._sampler.sample_and_replace_weights()
            if apply_sigmoid:
                predictions[sample] = tf.sigmoid(self._model.predict(data)).numpy()
            else:
                predictions[sample] = self._model.predict(data)
            self._model.set_weights(posterior_mean)
        return predictions