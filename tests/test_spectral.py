"""Tests to make sure the radiative module is working as expected."""

import cupy as cp
import numpy as np


def test_blackbody():
    from taurex.util.emission import black_body

    from taurex_cupy.spectral import cuda_blackbody

    temperature = np.linspace(100, 1000, 100)
    lamb = np.linspace(1, 10090, 100)

    results = np.array([black_body(lamb, t) for t in temperature])

    gpu_results = cuda_blackbody(cp.asarray(lamb), cp.asarray(temperature))

    np.testing.assert_allclose(results, gpu_results.get(), rtol=1e-5)
