"""Shape-preserving C-contiguous array conversion.

Unlike ``numpy.ascontiguousarray``, this helper preserves scalar shape ``()``.
"""

from __future__ import annotations

import numpy as np


def as_c_contiguous(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    if array.shape != np.shape(value):
        array = array.reshape(np.shape(value))
    return array


__all__ = ["as_c_contiguous"]
