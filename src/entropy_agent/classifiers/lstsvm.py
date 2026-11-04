"""One-vs-one LSTSVM, reusing the existing NumPy binary equations.

This is an OvO adapter, not a tree or the separate LST-KSVC algorithm.
The binary backend retains the repository's raw plane-residual comparison
and historical class-count-dependent RBF ridge factors (see audit document).
"""

from itertools import combinations

import numpy as np

from .bt_lstsvm import BinaryLSTSVM, _features


class LSTSVM:
    """Train K*(K-1)/2 independent binary models; vote with lowest-ID ties."""

    def __init__(self, *, c1=1.0, c2=1.0, kernel="rbf", bandwidth=1.0,
                 epsilon=1e-4):
        node = BinaryLSTSVM(c1=c1, c2=c2, kernel=kernel,
                            bandwidth=bandwidth, epsilon=epsilon)
        self.parameters = {name: getattr(node, name) for name in
                           ("c1", "c2", "kernel", "bandwidth", "epsilon")}

    def fit(self, x, y):
        features = _features(x)
        labels = np.asarray(y)
        if (labels.shape != (len(features),) or
                not np.issubdtype(labels.dtype, np.integer)):
            raise ValueError("multiclass labels must be a one-dimensional integer vector")
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("training requires at least two classes")
        pairs = tuple(combinations(range(len(classes)), 2))
        models = []
        for left, right in pairs:
            mask = (labels == classes[left]) | (labels == classes[right])
            binary_labels = np.where(labels[mask] == classes[left], 1, -1)
            node = BinaryLSTSVM(**self.parameters).fit(features[mask], binary_labels)
            if not all(np.isfinite(plane).all() for plane in (node._plane1, node._plane2)):
                raise ValueError("binary fit produced nonfinite planes")
            models.append(node)
        self.classes_ = classes.copy()
        self.n_features_in_ = features.shape[1]
        self.pairs_ = pairs
        self.models_ = models
        return self

    def predict_votes(self, x):
        if not hasattr(self, "models_"):
            raise ValueError("model has not been fitted")
        features = _features(x, width=self.n_features_in_)
        votes = np.zeros((len(features), len(self.classes_)), dtype=np.int64)
        for (left, right), node in zip(self.pairs_, self.models_, strict=True):
            winners = np.where(node.predict(features) == 1, left, right)
            votes[np.arange(len(features)), winners] += 1
        return votes

    def predict(self, x):
        votes = self.predict_votes(x)
        return self.classes_[np.argmax(votes, axis=1)]
