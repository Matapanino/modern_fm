"""Evaluation labels must belong to the training vocabulary before backend dispatch."""
import numpy as np
import pytest
from modern_fm import FFMClassifier, FMClassifier, FwFMClassifier


@pytest.mark.parametrize("cls", [FMClassifier, FFMClassifier, FwFMClassifier])
@pytest.mark.parametrize("labels,unknown,loss", [
    ([0, 2], 1, "logistic"), ([0, 2], 3, "logistic"),
    (["a", "c"], "b", "logistic"), ([0, 2, 4], 1, "logistic"),
    ([0, 2, 4], 5, "logistic"), ([0, 2], 1, "softmax"),
])
@pytest.mark.parametrize("as_list", [False, True])
def test_unknown_eval_class_rejected(cls, labels, unknown, loss, as_list):
    X = np.ones((12, 2))
    y = np.resize(labels, 12)
    model = cls(n_factors=2, max_iter=1, loss=loss, random_state=0)
    ev = (X[:2], np.asarray([labels[0], unknown]))
    if as_list:
        ev = [ev]
    kwargs = {} if cls is FMClassifier else {"field_ids": [0, 1]}
    with pytest.raises(ValueError, match="eval_set.*unknown class"):
        model.fit(X, y, eval_set=ev, **kwargs)


@pytest.mark.parametrize("cls", [FMClassifier, FFMClassifier, FwFMClassifier])
@pytest.mark.parametrize("labels", [[0, 2], ["a", "c"], [0, 2, 4]])
def test_eval_subset_and_column_targets_supported(cls, labels):
    X = np.ones((12, 2))
    kwargs = {} if cls is FMClassifier else {"field_ids": [0, 1]}
    model = cls(n_factors=2, max_iter=1, random_state=0)
    model.fit(X, np.resize(labels, 12), eval_set=(X[:2], [labels[-1]] * 2), **kwargs)
    expected = model.predict_proba(X)
    with pytest.warns(UserWarning):
        model.fit(X, np.resize(labels, 12),
                  eval_set=(X[:2], np.asarray([labels[-1]] * 2)[:, None]), **kwargs)
    np.testing.assert_array_equal(model.predict_proba(X), expected)


@pytest.mark.parametrize("cls", [FMClassifier, FFMClassifier, FwFMClassifier])
def test_eval_length_checked(cls):
    kwargs = {} if cls is FMClassifier else {"field_ids": [0, 1]}
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        cls(max_iter=1).fit(np.ones((4, 2)), [0, 1, 0, 1],
                            eval_set=(np.ones((2, 2)), [0]), **kwargs)
