from typing import List
import numpy as np
import sklearn.metrics


class Calibration:

    def __init__(self, precisions, confidences, counts, bins, acc_errors=None, counts_errors=None):
        self.precisions = precisions
        self.confidences = confidences
        self.counts = counts
        self.bins = bins
        self.avg_precision = np.sum(precisions * counts) / np.sum(counts)
        self.avg_confidence = np.sum(confidences * counts) / np.sum(counts)
        self.gaps = np.abs(precisions - confidences)
        self.ece = np.sum(self.gaps * counts) / np.sum(counts)
        self.mce = np.max(self.gaps)
        self.acc_erors = acc_errors
        self.counts_errors = counts_errors

    @classmethod
    def compute(cls,
                true_labels: np.ndarray,
                confidences: np.ndarray,
                nbins: int = 10,
                weights: List[float] = None) -> 'Calibration':

        if np.ndim(true_labels) == 1:
            true_labels = np.expand_dims(true_labels, axis=1)
            confidences = np.expand_dims(confidences, axis=1)

        if weights is None:
            weights = np.ones(true_labels.shape[1])

        bins = np.linspace(0.0, 1.0, nbins + 1)
        bin_precisions = np.zeros((nbins, len(weights)), dtype=np.float)
        bin_confidences = np.zeros((nbins, len(weights)), dtype=np.float)
        bin_counts = np.zeros((nbins, len(weights)), dtype=np.int)
        for w in range(len(weights)):
            indices = np.digitize(confidences[:, w], bins, right=True)
            for b in range(nbins):
                selected = np.where(indices == b + 1)[0]
                if len(selected) > 0:
                    bin_precisions[b, w] = len(np.where(indices * true_labels[:, w] == b + 1)[0]) / len(selected)
                    bin_confidences[b, w] = np.mean(confidences[:, w][selected])
                    bin_counts[b, w] = len(selected)

        avg_bin_precisions = np.average(bin_precisions, axis=1, weights=weights)
        avg_bin_confidences = np.average(bin_confidences, axis=1, weights=weights)
        avg_bin_counts = np.sum(bin_counts, axis=1)

        return cls(avg_bin_precisions, avg_bin_confidences, avg_bin_counts, bins)


class Auroc:

    @staticmethod
    def roc_curve(scores_id, scores_ood):
        labels_id = np.zeros(scores_id.shape[0])
        labels_ood = np.ones(scores_ood.shape[0])
        y_true = np.concatenate([labels_id, labels_ood])
        y_score = np.concatenate([scores_id, scores_ood])
        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true, y_score)

        return fpr, tpr, threshold

    @staticmethod
    def auroc(scores_id, scores_ood):
        labels_id = np.zeros(scores_id.shape[0])
        labels_ood = np.ones(scores_ood.shape[0])
        y_true = np.concatenate([labels_id, labels_ood])
        y_score = np.concatenate([scores_id, scores_ood])
        auroc = sklearn.metrics.roc_auc_score(y_true, y_score)

        return auroc

    @staticmethod
    def scores_for_positives(scores, confidences):
        func = np.min
        prediction = np.greater_equal(confidences, 0.5).astype(int)
        scores_pos = scores * prediction

        def func_nonzero(arr, func):
            nz = arr[arr != 0]
            return func(nz) if len(nz) > 0 else 0.0

        samplewise_mean_nz = np.apply_along_axis(func_nonzero, axis=1, arr=scores_pos, func=func)
        samplewise_mean_nz = samplewise_mean_nz[samplewise_mean_nz != 0]

        return samplewise_mean_nz
