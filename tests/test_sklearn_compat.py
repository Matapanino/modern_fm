"""scikit-learn estimator compliance via sklearn.utils.estimator_checks.

Runs the full ``check_estimator`` battery against the public estimators so they
behave as drop-in sklearn estimators (Pipeline / GridSearchCV / clone). Small,
fast configs keep the (~40 checks each) suite quick; n_jobs=1 keeps it
bit-deterministic for the idempotence checks.
"""

import pytest
from modern_fm import FFMClassifier, FFMRegressor, FMClassifier, FMRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

_ESTIMATORS = [
    FMClassifier(max_iter=20, n_factors=4, n_jobs=1, random_state=0),
    FMRegressor(max_iter=20, n_factors=4, n_jobs=1, random_state=0),
    FFMClassifier(max_iter=20, n_factors=4, n_jobs=1, random_state=0),
    FFMRegressor(max_iter=20, n_factors=4, n_jobs=1, random_state=0),
]

# These two checks assert sample_weight=w is identical to repeating a row w
# times. That holds for closed-form/full-batch losses but NOT for per-row SGD
# (the update order differs), so our gradient-trained FMs can't satisfy them.
_XFAIL = {
    "check_sample_weight_equivalence_on_dense_data",
    "check_sample_weight_equivalence_on_sparse_data",
}


def _check_name(check):
    return getattr(check, "__name__", None) or getattr(
        getattr(check, "func", None), "__name__", repr(check)
    )


@parametrize_with_checks(_ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    if _check_name(check) in _XFAIL:
        pytest.xfail("sample_weight != row-duplication under per-row SGD")
    check(estimator)
