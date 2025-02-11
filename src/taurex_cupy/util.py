"""utility functions for taurex-cupy"""

import cupy as cp
import numpy as np
import numpy.typing as npt


def cuda_find_closest_pair(arr: cp.ndarray, values: cp.ndarray) -> cp.ndarray:
    """
    Find the closest pair of values in an array

    Parameters
    ----------
    arr : cp.ndarray
        The array to search
    values : cp.ndarray
        The values to search for

    Returns
    -------
    cp.ndarray
        The indices of the closest pair
    """
    right = arr.searchsorted(values, side="right")
    right = cp.clip(right, 0, arr.shape[0] - 1)
    left = right - 1
    left = cp.clip(left, 0, arr.shape[0] - 1)

    return left, right


def determine_grid_slice(dest_wngrid: npt.NDArray[np.float64], src_wngrid: npt.NDArray[np.float64]) -> slice:
    """Determine the grid length of the destination grid."""
    min_grid_idx = 0
    max_grid_idx = None
    min_wn = dest_wngrid.min()
    max_wn = dest_wngrid.max()
    if min_wn is not None:
        min_grid_idx = max(np.argmax(min_wn < src_wngrid) - 1, 0)
    if max_wn is not None:
        max_grid_idx = np.argmax(src_wngrid >= max_wn) + 1

    return slice(min_grid_idx, max_grid_idx)
