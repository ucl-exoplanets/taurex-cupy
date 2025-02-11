"""Spectral functions for CUDA"""

import cupy as cp
import numpy as np
import numpy.typing as npt


def cuda_blackbody(lamb: npt.NDArray[np.floating], temperature: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Calculate the blackbody spectrum

    Parameters
    ----------
    lamb : npt.NDArray[np.floating]
        Wavelength grid
    temperature : npt.NDArray[np.floating]
        Temperature

    Returns
    -------
    npt.NDArray[np.floating]
        Blackbody spectrum
    """
    from taurex.constants import KBOLTZ as k
    from taurex.constants import PLANCK as h
    from taurex.constants import SPDLIGT as c

    temperature = cp.atleast_1d(temperature)
    lamb = cp.atleast_1d(lamb)

    temperature = temperature[:, None]
    lamb = lamb[None, :]

    lamb = 1e-2 / lamb
    planck = 2 * np.pi * h * c**2 / lamb**5
    exp = cp.expm1(h * c / (lamb * k * temperature))
    bb = planck / (exp)

    return bb * 1e-6
