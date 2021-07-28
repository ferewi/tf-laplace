# tf-laplace
Estimate the true posterior distribution over the weights of a 
neural network using the Kronecker Factored Laplace Approximation. 
Compatible with TensorFlow 2.x

[![Build Status](https://travis-ci.com/ferewi/tf-laplace.svg?branch=main)](https://travis-ci.com/ferewi/tf-laplace)

## Description
This library implements three Laplace approximation methods to 
approximate the true posterior distribution over the weights of 
a neural network. It is similar to this 
[implementation for PyTorch](https://github.com/DLR-RM/curvature) of [[2]](#ref2).

The library, located in the `laplace` package, implements three 
approximations to the Fisher Information Matrix:

1. Diagonal (DiagFisher) [[3]](#ref3)
2. Block-Diagonal (BlockDiagFisher)
3. KFAC (KFAC) [[1]](#ref1) [[4]](#ref4) [[5]](#ref5)

Furthermore, it implements a sampler for each approximation to 
sample weights from the approximated posterior distribution. The
library can be used with any Tensorflow 2 sequential model. So far, 
the approximations only considers Dense and Convolutional layers.

The `experiments` package contains code related to the experiments 
in `experiments.py`.  

## Install

To install the library, clone or download the repository and run:
```
pip install .
```
This will install all the following dependencies that are needed:
* numpy
* tensorflow (v2.5)
* tensorflow_probability

It will also install the following libraries that are needed to run
the demo in the provided jupyter notebook and the experiments in 
`experiments.py`:
* matplotlib
* pandas
* sklearn

## Usage
This mini-example as well as the demo provided in the jupyter notebook 
demonstrate how to use this library to approximate the posterior 
distribution of a neural network trained on a synthetic multi-label
classification dataset. This dataset is contained in the 
`experiments/dataset.py` module.

```python
# standard imports
import numpy as np
import tensorflow as tf

# library imports
from laplace.curvature import KFAC
from laplace.sampler import Sampler

# additional imports
from experiments.dataset import F3

# 1. Create Dataset
# We create a multi-label dataset which consists of two classes.
ds = F3.create(50, -5.5, 5.5)
training_set = tf.data.Dataset.from_tensor_slices(ds.get()).batch(32)
test_set = tf.data.Dataset.from_tensor_slices(ds.get_test_set(2000)).batch(256)

# 2. Build and Train Model
# As the method can be applied to already trained models this is 
# just for demo purposes.
NUM_CLASSES = 2
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=2, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(NUM_CLASSES)
])
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
model.fit(training_set, epochs=1000, verbose=True)

# 3. Approximate Curvature
# We approximate the curvature by running a training loop over 
# the training set. Instead of updating the models parameters 
# the curvature approximation gets computed incrementally.
kfac = KFAC.compute(model, training_set, criterion)
sampler = Sampler.create(kfac, tau=1, n=50)

# 4. Evaluate the Bayesian neural network on a Test Set
MC_SAMPLES = 50
predictions = np.zeros([MC_SAMPLES, 0, NUM_CLASSES], dtype=np.float32)
for i, (x, y) in enumerate(test_set):
    posterior_mean = model.get_weights()
    batch_predictions = np.zeros([MC_SAMPLES, x.shape[0], NUM_CLASSES], dtype=np.float32)
    for sample in range(MC_SAMPLES):
        sampler.sample_and_replace_weights()
        batch_predictions[sample] = tf.sigmoid(model.predict(x)).numpy()
        model.set_weights(posterior_mean)
    predictions = np.concatenate([predictions, batch_predictions], axis=1)
```

## Experiments
The experiments in `experiments.py` demonstrate how the model 
uncertainty estimates obtained from the Bayesian neural network 
can improve the calibration and out-of-distribution detection of
the deterministic model. All code related to the experiment can
be found in the `experiments` package.

### Calibration
The calibration experiment compares the calibration of the 
deterministic baseline model to that of the Bayesian neural 
network. This is visualised using calibration curve diagrams.

The experiment can be run via:
```shell
python experiments.py calibration
```

### Out-of_Distribution Detection
The out-of-distribution detection experiment demonstrates how 
using the predictive standard deviation of multiple probabilistic
forward passes improves the separability from in- and out-of-distribution
data compared to using the confidence scores of the baseline model. 
This is visualised using receiver operating characteristic (ROC) 
plots.

The experiment can be run via:
```shell
python experiments.py ood
```


## Bibliography

|   |   |
|---|---|
|[1]<a name="ref1"></a>|Ritter, H., Botev, A., & Barber, D. (2018, January). A scalable laplace approximation for neural networks. In 6th International Conference on Learning Representations, ICLR 2018-Conference Track Proceedings (Vol. 6). International Conference on Representation Learning.|
|[2]<a name="ref2"></a>|Lee, J., Humt, M., Feng, J., Triebel, R. (2020), Estimating Model Uncertainty of Neural Networks in Sparse Information Form. Proceedings of Machine Learning Research. International Conference on Machine Learning (ICML) |
|[3]<a name="ref3"></a>|Becker, S & Lecun, Y. (1988). Improving the convergence of back-propagation learning with second-order methods. In D. Touretzky, G. Hinton, & T. Sejnowski (Eds.), Proceedings of the 1988 Connectionist Models Summer School, San Mateo (pp. 29-37). Morgan Kaufmann.|
|[4]<a name="ref4"></a>|Martens, J., & Grosse, R. (2015, June). Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning (pp. 2408-2417).|
|[5]<a name="ref5"></a>|Botev, A., Ritter, H., & Barber, D. (2017, August). Practical gauss-newton optimisation for deep learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 557-565). JMLR. org.|