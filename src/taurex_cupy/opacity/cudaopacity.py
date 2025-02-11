"""Functions for calculating opacities on the GPU using CUDA"""

import typing as t

import cupy as cp
import numpy as np
import numpy.typing as npt
from taurex.cache import OpacityCache
from taurex.log import Logger
from taurex.opacity import InterpolatingOpacity, Opacity


def interpolate_cuda(
    xsec_grid: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    tgrid: npt.NDArray[np.float64],
    pgrid: npt.NDArray[np.float64],
    Tmin: npt.NDArray[np.int32],
    Tmax: npt.NDArray[np.int32],
    Pmin: npt.NDArray[np.int32],
    Pmax: npt.NDArray[np.int32],
    mix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    from taurex.util.math import interp_bilin_numpy

    x11 = xsec_grid[Pmin, Tmin]
    x12 = xsec_grid[Pmin, Tmax]
    x21 = xsec_grid[Pmax, Tmin]
    x22 = xsec_grid[Pmax, Tmax]

    temperature_min = tgrid[Tmin]
    temperature_max = tgrid[Tmax]
    pressure_min = pgrid[Pmin]
    pressure_max = pgrid[Pmax]

    return (
        interp_bilin_numpy(
            x11,
            x12,
            x21,
            x22,
            temperature[:, None],
            temperature_min[:, None],
            temperature_max[:, None],
            pressure[:, None],
            pressure_min[:, None],
            pressure_max[:, None],
        )
        * mix[:, None]
    )


class CudaOpacity(Logger):
    def __init__(self, molecule_name: str, wngrid: t.Optional[npt.NDArray[np.float64]] = None) -> None:
        super().__init__(self.__class__.__name__)
        self._xsec: InterpolatingOpacity = OpacityCache()[molecule_name]
        if self._xsec is None or not isinstance(self._xsec, Opacity):
            raise ValueError

        self._lenP = len(self._xsec.pressureGrid)
        self._lenT = len(self._xsec.temperatureGrid)
        self.info("Transfering xsec grid to GPU")
        self._gpu_tgrid = cp.array(self._xsec.temperatureGrid)
        self._gpu_pgrid = cp.array(np.log10(self._xsec.pressureGrid))

        self.transfer_xsec_grid(wngrid)

    def transfer_xsec_grid(self, wngrid: npt.NDArray[np.floating]) -> None:
        from ..mpi import gpu_single_allocation

        self._wngrid = self._xsec.wavenumberGrid
        xsecgrid = self._xsec.xsecGrid

        if wngrid is not None:
            from scipy.interpolate import interp1d

            self._wngrid = wngrid
            f = interp1d(
                self._xsec.wavenumberGrid,
                self._xsec.xsecGrid,
                copy=False,
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=True,
            )
            xsecgrid = f(wngrid).ravel().reshape(*xsecgrid.shape[0:-1], -1)  # Force contiguous array

        self._strides = xsecgrid.strides
        self._gpu_grid = gpu_single_allocation(xsecgrid, logger=self, owner=self)

    def opacity(
        self,
        temperature: npt.ArrayLike,
        pressure: npt.ArrayLike,
        mix: npt.ArrayLike,
        wngrid=None,
    ) -> cp.ndarray:
        from ..util import cuda_find_closest_pair, determine_grid_slice

        temperature = np.atleast_1d(temperature)
        pressure = np.atleast_1d(pressure)
        mix = np.atleast_1d(mix)

        minmaxT = self._xsec.temperatureGrid.min(), self._xsec.temperatureGrid.max()
        minmaxP = self._xsec.pressureGrid.min(), self._xsec.pressureGrid.max()
        temperature = np.clip(temperature, *minmaxT)
        pressure = np.clip(pressure, *minmaxP)

        gpu_grid = self._gpu_grid[...]
        gpu_tgrid = self._gpu_tgrid
        gpu_pgrid = self._gpu_pgrid

        if wngrid is not None:
            grid_slice = determine_grid_slice(wngrid, self._wngrid)
            gpu_grid = gpu_grid[..., grid_slice]

        temperature = cp.array(temperature)
        pressure = cp.log10(cp.array(pressure))
        t_min, t_max = cuda_find_closest_pair(gpu_tgrid, temperature)
        p_min, p_max = cuda_find_closest_pair(gpu_pgrid, pressure)

        mix = cp.array(mix)

        my_dest = interpolate_cuda(
            gpu_grid,
            temperature,
            pressure,
            gpu_tgrid,
            gpu_pgrid,
            t_min,
            t_max,
            p_min,
            p_max,
            mix,
        )

        return my_dest / 10000
