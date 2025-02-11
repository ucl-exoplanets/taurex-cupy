"""Test MPI GPU allocation"""


def test_gpu_shared():
    import numpy as np

    from taurex_cupy.mpi import gpu_single_allocation

    arr = np.random.rand(10)

    gpu_arr = gpu_single_allocation(arr)

    assert gpu_arr is not None
