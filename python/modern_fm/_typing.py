"""Shared type aliases."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

ArrayLike = np.ndarray
MatrixLike = np.ndarray | sp.csr_matrix | sp.csr_array
