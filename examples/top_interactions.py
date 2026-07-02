"""Inspecting the strongest learned pairwise interactions (docs/roadmap.md).

Fits an FM on synthetic data with one planted multiplicative interaction
(features 3 x 7 drive the label) and shows that `top_interactions` surfaces
that pair from the learned factors. The strength is the magnitude of the
learned pairwise coefficient of `x_i x_j` — |<V_i, V_j>| for FM
(docs/math_spec.md).

    .venv/bin/python examples/top_interactions.py
"""

import numpy as np
from modern_fm import FMClassifier


def main():
    rng = np.random.default_rng(0)
    n, d = 4000, 12
    X = rng.normal(size=(n, d))
    # the label depends almost entirely on the product x_3 * x_7
    y = (3.0 * X[:, 3] * X[:, 7] + 0.3 * rng.normal(size=n) > 0).astype(int)

    model = FMClassifier(n_factors=8, max_iter=40, learning_rate=0.1, random_state=0)
    model.fit(X, y)

    print("top learned pairwise interactions (planted pair: 3 x 7):")
    print(f"{'rank':>5} {'pair':>10} {'strength':>10}")
    for rank, (i, j, s) in enumerate(model.top_interactions(5), start=1):
        print(f"{rank:>5} {f'({i}, {j})':>10} {s:>10.4f}")


if __name__ == "__main__":
    main()
