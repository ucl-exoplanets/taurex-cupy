"""Test fdunction that contribute to optical depth."""

import cupy as cp
import numpy as np
import numpy.typing as npt


def contribute_tau_test(
    startk: int,
    endk: int,
    density_offset: int,
    sigma: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    nlayers: int,
    ngrid: int,
    layer: int,
    tau: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    _path = 1.0
    _density = 1.0
    _path = path[startk:endk, None]
    _density = density[startk + density_offset : endk + density_offset, None]
    _sigma = sigma[startk + layer : endk + layer, :]

    tau[layer, :] += np.sum(_sigma * _path * _density, axis=0)

    return tau


def test_contribute_cuda():
    """Tests the numpy version of the contribution function (eclipse version)."""
    from taurex_cupy.contributions.cudacontrib import cuda_contribute_tau

    NLAYERS = 200
    WNGRID = 1000
    sigma = np.random.rand(NLAYERS, WNGRID)
    density = np.random.rand(NLAYERS)
    path = np.random.rand(NLAYERS, NLAYERS)
    tau = np.zeros((NLAYERS, WNGRID))
    start_k = np.arange(NLAYERS)
    end_k = start_k + 1
    res = np.array([
        contribute_tau_test(sk, ek, 0, sigma, density, p, NLAYERS, WNGRID, 0, tau[idx : idx + 1])
        for idx, (sk, ek, p) in enumerate(zip(start_k, end_k, path))
    ]).reshape((NLAYERS, WNGRID))

    sigma_gpu = cp.asarray(sigma)
    density_gpu = cp.asarray(density)
    tau_gpu = cp.zeros((NLAYERS, WNGRID))
    path_gpu = cp.asarray(path)
    start_k_gpu = cp.asarray(start_k)
    end_k_gpu = cp.asarray(end_k)
    cuda_contribute_tau(
        start_k_gpu,
        end_k_gpu,
        cp.zeros(NLAYERS),
        sigma_gpu,
        density_gpu,
        path_gpu,
        NLAYERS,
        WNGRID,
        tau_gpu,
        start_layer=0,
        total_layers=NLAYERS,
        with_layer_offset=False,
    )

    np.testing.assert_allclose(res, tau_gpu.get())


def test_contribute_cuda_offset():
    """Tests the numpy version of the contribution function (transmission version)."""
    from taurex_cupy.contributions.cudacontrib import cuda_contribute_tau

    NLAYERS = 200
    WNGRID = 1000
    sigma = np.random.rand(NLAYERS, WNGRID)
    density = np.random.rand(NLAYERS)
    path = np.random.rand(NLAYERS, NLAYERS)
    tau = np.zeros((NLAYERS, WNGRID))
    start_k = np.arange(NLAYERS)
    end_k = NLAYERS - start_k
    [
        contribute_tau_test(sk, ek, 0, sigma, density, p, NLAYERS, WNGRID, idx, tau)
        for idx, (sk, ek, p) in enumerate(zip(start_k, end_k, path))
    ]

    sigma_gpu = cp.asarray(sigma)
    density_gpu = cp.asarray(density)
    tau_gpu = cp.zeros((NLAYERS, WNGRID))
    path_gpu = cp.asarray(path)
    start_k_gpu = cp.asarray(start_k)
    end_k_gpu = cp.asarray(end_k)
    cuda_contribute_tau(
        start_k_gpu,
        end_k_gpu,
        cp.zeros(NLAYERS),
        sigma_gpu,
        density_gpu,
        path_gpu,
        NLAYERS,
        WNGRID,
        tau_gpu,
        start_layer=0,
        total_layers=NLAYERS,
        with_layer_offset=True,
    )

    np.testing.assert_allclose(tau, tau_gpu.get())
