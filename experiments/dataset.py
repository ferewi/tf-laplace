from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import tensorflow as tf


class _BaseDataset(ABC):
    """
    Abstract base class for datasets.

    Attributes:
        classes (list): The classes belonging to the dataset.
        samples (list): All samples of the dataset.
    """

    def __init__(self, name: str):
        self.name = name
        self.classes = []
        self.samples = []

    def add_class(self, c):
        c.set_dataset(self)
        self.classes.append(c)

    def add_sample(self, sample):
        self.samples.append(sample)

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the Datasets in data (X) and labels (y).

        Returns:
            Tuple[np.ndarray, np.ndarray]
        """
        data = np.ndarray((len(self.samples), 2), dtype=np.float32)
        labels = np.zeros((len(self.samples), len(self.classes)), dtype=np.int)
        for i, sample in enumerate(self.samples):
            data[i] = [sample.x, sample.y]
            for label in sample.labels:
                labels[i, label-1] = 1
        return data, labels

    def as_tf_dataset(self) -> tf.data.Dataset:
        data, labels = self.get()
        return tf.data.Dataset.from_tensor_slices((data, labels))

    @abstractmethod
    def load_labels(self):
        pass

    def _sanitize_size(self, n):
        # sanitize dataset size
        for i in range(len(self.samples), n, -1):
            self.samples.pop()

    @classmethod
    @abstractmethod
    def create(cls, **kwargs):
        pass


class _BasicClass(ABC):
    def __init__(self, label):
        self.label = label
        self.dataset = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def add_sample(self, sample):
        sample.add_to_class(self.label)

    @abstractmethod
    def sample(self, n):
        pass

    @abstractmethod
    def test_sample_for_class(self, sample):
        pass

    @abstractmethod
    def print(self, ax):
        pass


class Sample:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.labels = []
        self.label_code = 0

    def add_to_class(self, c):
        if not self.has_label(c):
            self.labels.append(c)

    def has_label(self, label):
        return label in self.labels


class F3(_BaseDataset):
    """
    A F3 Datasets consists of points sampled along f(x) = x**3 with x in [lb, ub].

    Attributes:
        lb (float): Lower bound for x coordinates.
        ub (float): Upper bound for x coordinates.
    """

    def __init__(self, lb, ub):
        super().__init__("F3")
        self.lb = lb
        self.ub = ub
        self.test_set = None
        self.validation_set = None
        self.ood_set = None

    @classmethod
    def create(cls, n: int, lb: float, ub: float) -> 'F3':
        """
        Creates a "Cubic Function" dataset with n samples having x coordinates between lb and ub.

        Args:
            n:  Number of samples.
            lb: Lower bound for x coordinates.
            ub: Upper bound for x coordinates.

        Returns:
            A F3 dataset.
        """
        ds = cls(lb, ub)
        ds.add_class(CubicClass(3, cls._test_class_above, 1))
        ds.add_class(CubicClass(3, cls._test_class_below, 2))

        points_x = np.random.uniform(ds.lb, ds.ub, n)
        points_y = np.random.normal(points_x ** 3, 3 ** 3, n)

        points = np.array(list(zip(points_x, points_y)))
        for point in points:
            sample = Sample(point[0], point[1])
            ds.add_sample(sample)
            for c in ds.classes:
                if not sample.has_label(c):
                    if c.test_sample_for_class(sample):
                        c.add_sample(sample)

        ds.get_validation_set()
        ds.get_test_set()
        ds.get_ood_set()
        return ds

    def load_labels(self):
        return ["red", "blue"]

    def get_training_set(self):
        return self.get()

    def get_validation_set(self, n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        if self.validation_set is None and n > 0:
            x = np.random.uniform(self.lb, self.ub, n)
            y = np.random.uniform(self.lb**3, self.ub**3, n)
            labels = np.zeros([n, 2], dtype=np.int)
            labels[:, 0] = self._test_class_above(x, y)
            labels[:, 1] = self._test_class_below(x, y)
            self.test_set = (np.stack((x, y), axis=1), labels)

        return self.validation_set

    def get_test_set(self, n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        if self.test_set is None and n > 0:
            x = np.random.uniform(self.lb, self.ub, n)
            y = np.random.uniform(self.lb**3, self.ub**3, n)
            labels = np.zeros([n, 2], dtype=np.int)
            labels[:, 0] = self._test_class_above(x, y)
            labels[:, 1] = self._test_class_below(x, y)
            self.test_set = (np.stack((x, y), axis=1), labels)

        return self.test_set

    def get_ood_set(self, n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        if self.ood_set is None and n > 0:
            points_x_l = np.random.uniform(-10, 0, int(n / 2))
            points_y_l = np.random.uniform(-400, self.lb**3, int(n / 2))
            points_x_r = np.random.uniform(0, 10, int(n / 2))
            points_y_r = np.random.uniform(self.ub**3, 400, int(n / 2))
            x = np.concatenate([points_x_l, points_x_r])
            y = np.concatenate([points_y_l, points_y_r])
            labels = np.zeros([n, 2], dtype=np.int)
            labels[:, 0] = self._test_class_above(x, y)
            labels[:, 1] = self._test_class_below(x, y)
            self.ood_set = (np.stack((x, y), axis=1), labels)

        return self.ood_set

    @staticmethod
    def _test_class_above(x, y):
        return y > x ** 3 + -10

    @staticmethod
    def _test_class_below(x, y):
        return y < x ** 3 + 10


class CubicClass(_BasicClass):
    def __init__(self, exponent, criterion, label):
        super().__init__(label)
        self.exponent = exponent
        self.criterion = criterion

    def sample(self, n):
        pass

    def test_sample_for_class(self, sample):
        return self.criterion(sample.x, sample.y)

    def print(self, ax):
        fx = np.linspace(self.dataset.lb, self.dataset.ub, num=100)
        fy = fx ** 3
        ax.plot(fx, fy, color='black')
