import cupy as cp
import numpy as np
import numpy.typing as npt
from taurex.cache import CIACache
from taurex.log import Logger


def interpolate_cuda(
    xsec_grid: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    tgrid: npt.NDArray[np.float64],
    Tmin: npt.NDArray[np.int32],
    Tmax: npt.NDArray[np.int32],
    mix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    from taurex.util.math import interp_lin_numpy

    x11 = xsec_grid[Tmin]
    x12 = xsec_grid[Tmax]
    temperature_min = tgrid[Tmin]
    temperature_max = tgrid[Tmax]

    return (
        interp_lin_numpy(
            x11,
            x12,
            temperature[:, None],
            temperature_min[:, None],
            temperature_max[:, None],
        )
        * mix[:, None]
    )


class CudaCIA(Logger):
    def __init__(self, pair_name, wngrid=None):
        super().__init__(self.__class__.__name__)
        self._xsec = CIACache()[pair_name]
        self._gpu_tgrid = cp.asarray(self._xsec.temperatureGrid)
        self.transfer_xsec_grid(wngrid)

    def transfer_xsec_grid(self, wngrid):
        self._wngrid = self._xsec.wavenumberGrid
        xsecgrid = self._xsec._xsec_grid

        if wngrid is not None:
            from scipy.interpolate import interp1d

            self._wngrid = wngrid
            f = interp1d(
                self._xsec.wavenumberGrid,
                xsecgrid,
                copy=False,
                bounds_error=False,
                fill_value=1e-60,
                assume_sorted=True,
            )
            xsecgrid = f(wngrid).ravel().reshape(*xsecgrid.shape[0:-1], -1)  # Force contiguous array

        self._strides = xsecgrid.strides
        self._gpu_grid = cp.array(xsecgrid)

    def opacity(self, temperature, mix, wngrid=None):
        from ..util import cuda_find_closest_pair, determine_grid_slice

        temperature = np.atleast_1d(temperature)
        mix = np.atleast_1d(mix)

        temperature = cp.asarray(temperature)
        mix = cp.asarray(mix)

        gpu_grid = self._gpu_grid[...]
        if wngrid is not None:
            grid_slice = determine_grid_slice(wngrid, self._wngrid)
            gpu_grid = gpu_grid[..., grid_slice]

        t_min, t_max = cuda_find_closest_pair(self._gpu_tgrid, temperature)

        return interpolate_cuda(gpu_grid, temperature, self._gpu_tgrid, t_min, t_max, mix)
