import typing as t

import cupy as cp
import cupyx as cpx
import numpy as np
import numpy.typing as npt
from taurex.chemistry import Chemistry
from taurex.contributions import Contribution
from taurex.model import OneDForwardModel
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile

from taurex_cupy.contributions.cudacontrib import CudaContribution


class TransmissionCudaModel(OneDForwardModel):
    """A forward model for transits using GPU acceleration."""

    def __init__(
        self,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: t.Optional[int] = 100,
        atm_min_pressure: t.Optional[float] = 1e-4,
        atm_max_pressure: t.Optional[float] = 1e6,
        contributions: t.Optional[list[Contribution | CudaContribution]] = None,
    ):
        """Initialise the model.

        Args:
            planet: Planet object
            star: Star object
            pressure_profile: Pressure profile for the atmosphere
            temperature_profile: Temperature profile for the atmosphere
            chemistry: Chemical model
            nlayers: Number of layers
            atm_min_pressure: Minimum pressure (If pressure profile is not set)
            atm_max_pressure: Maximum pressure (If pressure profile is not set)
            contributions: List of contributions

        """
        super().__init__(
            name=self.__class__.__name__,
            planet=planet,
            star=star,
            pressure_profile=pressure_profile,
            temperature_profile=temperature_profile,
            chemistry=chemistry,
            nlayers=nlayers,
            atm_min_pressure=atm_min_pressure,
            atm_max_pressure=atm_max_pressure,
            contributions=contributions,
        )

    def compute_path_length(self, dz: npt.NDArray[np.floating]) -> list[npt.NDArray[np.float64]]:
        r"""Compute path length for each layer, new method.

        Args:
            dz: $\Delta z$ of the layer (altitude)

        Returns:
            list: Path length for each layer

        """
        from taurex.util.geometry import parallel_vector

        altitude_boundaries = self.altitude_boundaries
        radius = self.planet.fullRadius

        # Generate our line of sight paths
        viewer, tangent = parallel_vector(radius, self.altitude_profile + dz / 2, altitude_boundaries.max())

        path_lengths = self.planet.compute_path_length(altitude_boundaries, viewer, tangent)
        # We need to pad the path lengths to the number of layers
        # path_lengths = [l for _, l in path_lengths]
        dls = []
        for _, p in path_lengths:
            # We need to pad the path lengths to the number of layers
            p = np.pad(p, (0, self.nLayers - len(p)), "constant", constant_values=0)
            dls.append(p)

        return dls
        #

    @property
    def cuda_contributions(self) -> None:
        """Get contributions that use cuda."""
        return [c for c in self.contribution_list if isinstance(c, CudaContribution)]

    @property
    def non_cuda_contributions(self) -> None:
        """Get contributions that do not use cuda."""
        return [c for c in self.contribution_list if not isinstance(c, CudaContribution)]

    def build(self) -> None:
        """Build the model."""
        super().build()
        for contrib in self.contribution_list:
            contrib.build(self)
        self._startK = cp.array(np.array([0 for x in range(self.nLayers)]).astype(np.int32))
        self._endK = cp.array(np.array([self.nLayers - x for x in range(self.nLayers)]).astype(np.int32))
        self._density_offset = cp.array(np.array(list(range(self.nLayers))).astype(np.int32))

        # self._tau_buffer= drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1], self.nLayers,),dtype=np.float64)

    def path_integral(
        self, wngrid: npt.NDArray[np.floating], return_contrib: bool
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        r"""Compute the path integral for the model.

        Args:
            wngrid: Wavenumber grid
            return_contrib: Return contributions

        Returns:
            tuple: $(R_p/R_s)^2$ and optical depth

        """
        total_layers = self.nLayers

        dz = self.deltaz

        wngrid_size = wngrid.shape[0]
        self._ngrid = wngrid_size
        cpu_dl = self.compute_path_length(dz)
        gpu_dl = cp.array(cpu_dl)
        density_profile = cp.array(self.densityProfile)

        self._fully_cuda = len(self.non_cuda_contributions) == 0

        tau = cp.zeros(
            shape=(total_layers, wngrid_size),
            dtype=np.float64,
        )

        tau_host = cpx.zeros_pinned(shape=(total_layers, wngrid_size), dtype=np.float64)
        if not self._fully_cuda:
            tau.set(self.fallback_noncuda(total_layers, cpu_dl, self.densityProfile, dz))

        for contrib in self.cuda_contributions:
            contrib.contribute(
                self,
                self._startK,
                self._endK,
                self._density_offset,
                0,
                density_profile,
                tau,
                path_length=gpu_dl,
            )

        rprs, tau = self.compute_absorption(tau, cp.array(dz))
        tau.get(out=tau_host)
        # cp.cuda.runtime.deviceSynchronize()
        final_rprs = rprs.get()

        return final_rprs, tau_host

    def fallback_noncuda(
        self,
        total_layers: int,
        path_length: npt.NDArray[np.floating],
        density_profile: npt.NDArray[np.floating],
        dz: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Fallback for non-cuda contributions.

        This will compute them on the CPU before copying them to the GPU.

        Args:
            total_layers: Total layers
            path_length: Path length
            density_profile: Density profile
            dz: Delta altitude of the layer

        Returns:
            Optical depth

        """
        tau = np.zeros(shape=(total_layers, self._ngrid))
        for layer in range(total_layers):
            self.debug("Computing layer %s", layer)
            dl = path_length[layer]

            endK = total_layers - layer

            for contrib in self.non_cuda_contributions:
                self.debug("Adding contribution from %s", contrib.name)
                contrib.contribute(self, 0, endK, layer, layer, density_profile, tau, path_length=dl)
        return tau

    def compute_absorption(self, tau: cp.ndarray, dz: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        r"""Compute the absorption.

        Args:
            tau: Tau
            dz: $\Delta z$ of the layer (altitude)

        Returns:
            tuple: $(R_p/R_s)^2$ and tau

        """
        cp.exp(-tau, out=tau)
        ap = cp.array(self.altitudeProfile[:, None])
        pradius = self._planet.fullRadius
        sradius = self._star.radius
        _dz = dz[:, None]

        integral = cp.sum((pradius + ap) * (1.0 - tau) * _dz * 2.0, axis=0)
        return ((pradius * pradius) + integral) / (sradius**2), tau

    @classmethod
    def input_keywords(cls):
        return [
            "transmission_cuda",
            "transit_cuda",
        ]
