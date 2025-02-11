"""Function for MPI related operations."""

import os
import typing as t

import cupy as cp
import numpy as np
from cupy.cuda.memory import MemoryPointer, UnownedMemory
from cupy.cuda.runtime import ipcGetMemHandle, ipcOpenMemHandle
from taurex.cache import GlobalCache
from taurex.log import Logger, setup_log
from taurex.mpi import shared_comm, shared_rank

_log = setup_log(__name__)


def gpu_single_allocation(
    arr: np.ndarray,
    logger: t.Optional[Logger] = None,
    force_shared: bool = False,
    owner: object = None,
) -> cp.ndarray:
    """Allocate the array on the GPU ensuring that it is only allocated once.

    This will only allocate the memory once multiple processes are bound to the same GPU.


    Args:
        arr (np.ndarray): The array to allocate on the GPU
        logger (t.Optional[Logger], optional): The logger to use. Defaults to None.
        force_shared (bool, optional): Force the allocation to be shared. Defaults to False.

    """
    import importlib.util

    has_mpi = importlib.util.find_spec("mpi4py") is not None
    if not has_mpi:
        return cp.asarray(arr)

    g_cache = GlobalCache()

    logger = logger or _log

    shared_memory_enabled = g_cache["gpu_shared_memory"] or g_cache["gpu_single_allocate"] or force_shared

    if not shared_memory_enabled:
        return cp.asarray(arr)

    _log.info("Using shared memory allocation")
    _log.info("Moving to GPU once")
    comm = shared_comm()

    cuda_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    myrank = shared_rank()

    cuda_comm = comm.Split(cuda_device, myrank)

    cuda_rank = cuda_comm.Get_rank()

    my_array = None
    shape, dtype, size, strides = arr.shape, arr.dtype, arr.size, arr.strides
    h = None
    if cuda_rank == 0:
        my_array = cp.asarray(arr)
        # Get ipc handle
        h = ipcGetMemHandle(my_array.data.ptr)

    h, shape, dtype, size, strides = cuda_comm.bcast((h, shape, dtype, size, strides))

    if cuda_rank != 0:
        arr_ptr = ipcOpenMemHandle(h)
        mem = UnownedMemory(arr_ptr, size, owner=owner)
        memptr = MemoryPointer(mem, offset=0)
        arr = cp.ndarray(shape=shape, dtype=dtype, memptr=memptr, strides=strides)
    return arr
