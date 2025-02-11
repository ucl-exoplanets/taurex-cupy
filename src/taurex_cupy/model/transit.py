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
    """

    A forward model for transits using GPU acceleration

    Parameters
    ----------

    planet: :class:`~taurex.data.planet.Planet`, optional
        Planet model, default planet is Jupiter

    star: :class:`~taurex.data.stellar.star.Star`, optional
        Star model, default star is Sun-like

    pressure_profile: :class:`~taurex.data.profiles.pressure.pressureprofile.PressureProfile`, optional
        Pressure model, alternative is to set ``nlayers``, ``atm_min_pressure``
        and ``atm_max_pressure``

    temperature_profile: :class:`~taurex.data.profiles.temperature.tprofile.TemperatureProfile`, optional
        Temperature model, default is an :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
        profile at 1500 K

    chemistry: :class:`~taurex.data.profiles.chemistry.chemistry.Chemistry`, optional
        Chemistry model, default is
        :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry` with
        ``H2O`` and ``CH4``

    nlayers: int, optional
        Number of layers. Used if ``pressure_profile`` is not defined.

    atm_min_pressure: float, optional
        Pressure at TOA. Used if ``pressure_profile`` is not defined.

    atm_max_pressure: float, optional
        Pressure at BOA. Used if ``pressure_profile`` is not defined.

    num_streams: int, optional
        Non-functional for now.

    """

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

    def compute_path_length(self, dz) -> list[npt.NDArray[np.float64]]:
        """Compute path length for each layer, new method."""
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
    def cuda_contributions(self):
        return [c for c in self.contribution_list if isinstance(c, CudaContribution)]

    @property
    def non_cuda_contributions(self):
        return [c for c in self.contribution_list if not isinstance(c, CudaContribution)]

    def build(self):
        super().build()
        self._startK = cp.array(np.array([0 for x in range(self.nLayers)]).astype(np.int32))
        self._endK = cp.array(np.array([self.nLayers - x for x in range(self.nLayers)]).astype(np.int32))
        self._density_offset = cp.array(np.array(list(range(self.nLayers))).astype(np.int32))
        # self._tau_buffer= drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1], self.nLayers,),dtype=np.float64)

    def path_integral(self, wngrid: npt.NDArray[np.floating], return_contrib: bool):
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
            tau[...] = cp.array(self.fallback_noncuda(total_layers, cpu_dl, self.densityProfile, dz))

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

        final_rprs = rprs.get()
        final_tau = np.copy(tau_host)

        return final_rprs, final_tau

    def fallback_noncuda(self, total_layers, path_length, density_profile, dz):
        tau = np.zeros(shape=(total_layers, self._ngrid))
        for layer in range(total_layers):
            self.debug("Computing layer %s", layer)
            dl = path_length[layer]

            endK = total_layers - layer

            for contrib in self.non_cuda_contributions:
                self.debug("Adding contribution from %s", contrib.name)
                contrib.contribute(self, 0, endK, layer, layer, density_profile, tau, path_length=dl)
        return tau

    def compute_absorption(self, tau, dz):
        tau = cp.exp(-tau)
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
