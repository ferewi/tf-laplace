import random
import sys
import numpy as np
import tensorflow as tf

from laplace.curvature import KFAC
from laplace.sampler import Sampler

from experiments.dataset import F3
from experiments.metrics import Calibration
from experiments.plots import PlotSpec, ReliabilityGraphSpec, ReliabilityDiagramPlotter


def build_and_train_model(training_set, optimizer, loss):
    NUM_CLASSES = 2
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_dim=2, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.fit(training_set, epochs=1000, verbose=True)

    return model


def eval_baseline(model, test_set):
    points = tf.zeros((0, 2), dtype=tf.float32)
    true_labels = tf.zeros([0, 2], dtype=tf.int64)
    predictions = tf.zeros([0, 2], dtype=tf.float32)
    for i, (data, y_true) in enumerate(test_set):
        output = model.predict_on_batch(x=data)
        prediction = tf.nn.sigmoid(output)
        points = tf.concat([points, tf.cast(data, dtype=tf.float32)], axis=0)
        true_labels = tf.concat([true_labels, tf.cast(y_true, tf.int64)], axis=0)
        predictions = tf.concat([predictions, prediction], axis=0)

    return points, true_labels, predictions


def eval_bnn(model, test_set, sampler, num_samples=10):
    predictions = np.zeros([num_samples, 0, 2], dtype=np.float32)
    for i, (x, y) in enumerate(test_set):
        posterior_mean = model.get_weights()
        batch_predictions = np.zeros([num_samples, x.shape[0], 2], dtype=np.float32)
        for sample in range(num_samples):
            sampler.sample_and_replace_weights()
            batch_predictions[sample] = tf.sigmoid(model.predict(x)).numpy()
            model.set_weights(posterior_mean)
        predictions = np.concatenate([predictions, batch_predictions], axis=1)

    return predictions


def eval_calibration(true_labels, baseline_predictions, bnn_predictions):
    calibration_bl = Calibration.compute(true_labels.numpy(), baseline_predictions.numpy(), nbins=10)
    calibration_bnn = Calibration.compute(true_labels.numpy(), bnn_predictions.mean(axis=0), nbins=10)

    plotspec = PlotSpec()
    plotspec.add(ReliabilityGraphSpec(calibration_bl, netarch="Mlp2", nettype='basline'))
    plotspec.add(ReliabilityGraphSpec(calibration_bnn, netarch="Mlp2", nettype='bnn'))

    ReliabilityDiagramPlotter.from_plot_spec(plotspec, with_histogram=True, with_ece=True,
                                             ylabel=r"$Precision_{macro}$")

def calibration():
    # 1. create dataset
    ds = F3.create(50, -5.5, 5.5)
    training_set = tf.data.Dataset.from_tensor_slices(ds.get()).batch(32)
    test_set = tf.data.Dataset.from_tensor_slices(ds.get_test_set(2000)).batch(256)

    # 2. build and train baseline model
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = build_and_train_model(training_set, optimizer, loss)

    # 3. approximate curvature
    kfac = KFAC.compute(model, training_set, loss)
    sampler = Sampler.create(kfac, tau=10, n=100)

    # 4. evaluate baseline and bayesian neural network
    points, true_labels, baseline_predictions = eval_baseline(model, test_set)
    bnn_predictions = eval_bnn(model, test_set, sampler, num_samples=50)

    # 5. evaluate calibration
    eval_calibration(true_labels, baseline_predictions, bnn_predictions)


def ood_detection():
    pass


def help():
    print("Error! Please provide the experiment you want to run as an argument:\n"
          "   python experiment/experiments.py calibration\n"
          "   python experiment/experiments.py ood\n")

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    tf.random.set_seed(666)

    if len(sys.argv) != 2:
        help()
        exit(1)
    elif sys.argv[1] == 'calibration':
        calibration()
    elif sys.argv[1] == 'ood':
        ood_detection()
    else:
        help()
        exit(1)
