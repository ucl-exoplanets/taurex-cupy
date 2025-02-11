"""test utils"""

import cupy as cp


def test_cuda_find_closest_pair():
    import numpy as np
    from taurex.util import find_closest_pair

    from taurex_cupy.util import cuda_find_closest_pair

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    values = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])

    taurex_cpu_left, taurex_cpu_right = list(zip(*[find_closest_pair(arr, v) for v in values]))
    taurex_cpu_left = np.array(taurex_cpu_left)
    taurex_cpu_right = np.array(taurex_cpu_right)

    taurex_gpu_left, taurex_gpu_right = cuda_find_closest_pair(cp.array(arr), cp.array(values))

    np.testing.assert_equal(taurex_cpu_left, taurex_gpu_left.get())
    np.testing.assert_equal(taurex_cpu_right, taurex_gpu_right.get())
