import math
from functools import lru_cache

import cupy as cp
import cupyx as cpx
import numpy as np
from taurex.model.simplemodel import SimpleForwardModel

from ..contributions import CudaContribution
from ..spectral import cuda_blackbody


@lru_cache(maxsize=400)
def gen_partial_kernal(ngauss, nlayers, grid_size):
    from taurex.constants import PI

    mu, weight = np.polynomial.legendre.leggauss(ngauss)
    mu_quads = (mu + 1) / 2

    code = f"""

    extern "C" __global__ void quadrature_kernal(double* __restrict__ dest,
                                        double* __restrict__ layer_tau,
                                        const double* __restrict__ dtau,
                                        const double* __restrict__ BB)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( i >= {grid_size} )
        return;
    """
    # Surface term
    for idx, mu in enumerate(mu_quads):
        code += f"""

            double I_{idx} = exp(-{1.0 / mu}*dtau[i])*BB[i]*{1.0 / PI};

        """

    code += f"""



        for (int layer = 0; layer < {nlayers}; layer++)
        {{

            double _dtau = dtau[layer*{grid_size} + i];
            double _layer_tau = layer_tau[layer*{grid_size} + i];
            double _BB = BB[layer*{grid_size} + i]*{1.0 / PI};
            layer_tau[layer*{grid_size} + i] = exp(-_layer_tau) - exp(-_dtau);
            _dtau += _layer_tau;

        """
    for idx, mu in enumerate(mu_quads):
        code += f"""

            I_{idx} += (exp(-_layer_tau*{1.0 / mu}) - exp(-_dtau*{1.0 / mu}))*_BB;
        """

    code += """
            }

        """

    for idx, _ in enumerate(mu_quads):
        code += f"""
            dest[{idx * grid_size}+i] = I_{idx};

        """

    code += """
    }
    """

    return cp.RawKernel(code, "quadrature_kernal")


class EmissionCudaModel(SimpleForwardModel):
    """

    A forward model for eclipse models using CUDA

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

    ngauss: int, optional
        Number of Gaussian quadrature points. Default is 4

    """

    def __init__(
        self,
        planet=None,
        star=None,
        pressure_profile=None,
        temperature_profile=None,
        chemistry=None,
        nlayers=100,
        atm_min_pressure=1e-4,
        atm_max_pressure=1e6,
        ngauss=4,
    ):
        super().__init__(
            self.__class__.__name__,
            planet,
            star,
            pressure_profile,
            temperature_profile,
            chemistry,
            nlayers,
            atm_min_pressure,
            atm_max_pressure,
        )

        self.set_num_gauss(ngauss)

    def set_num_gauss(self, value):
        self._ngauss = int(value)
        mu, weight = np.polynomial.legendre.leggauss(self._ngauss)
        self._mu_quads = (mu + 1) / 2
        self._wi_quads = (weight) / 2

        self._gpu_mu_quads = cp.array(self._mu_quads)
        self._gpu_wi_quads = cp.array(self._wi_quads)

    def build(self):
        super().build()
        self._start_surface_K = cp.array(np.array([0]).astype(np.int32))
        self._end_surface_K = cp.array(np.array([self.nLayers]).astype(np.int32))

        self._start_layer = cp.arange(self.nLayers, dtype=np.int32) + 1
        self._end_layer = cp.full_like(self._start_layer, self.nLayers)

        self._start_dtau = cp.arange(self.nLayers)
        self._end_dtau = self._start_dtau + 1

        self._dz = cp.zeros(
            shape=(
                self.nLayers,
                self.nLayers,
            ),
            dtype=np.float64,
        )
        self._density_offset = cp.zeros(shape=(self.nLayers,), dtype=np.int32)

        # self._tau_buffer= drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1], self.nLayers,),dtype=np.float64)

    def partial_model(self, wngrid=None, cutoff_grid=True):
        from taurex.util.util import clip_native_to_wngrid

        self.initialize_profiles()

        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)
        self._star.initialize(native_grid)

        for contrib in self.contribution_list:
            contrib.prepare(self, native_grid)

        return self.evaluate_emission(native_grid, False)

    @property
    def cuda_contributions(self):
        return [c for c in self.contribution_list if isinstance(c, CudaContribution)]

    @property
    def non_cuda_contributions(self):
        return [c for c in self.contribution_list if not isinstance(c, CudaContribution)]

    def evaluate_emission(self, wngrid, return_contrib, keep_in_gpu=False):
        total_layers = self.nLayers

        dz = self.deltaz

        dz = np.array([dz for x in range(self.nLayers)])
        self._dz = cp.array(dz)

        wngrid_size = wngrid.shape[0]
        temperature = self.temperatureProfile
        density_profile = cp.array(self.densityProfile)

        self._fully_cuda = len(self.non_cuda_contributions) == 0

        layer_tau = cp.zeros(
            shape=(total_layers, wngrid_size),
            dtype=np.float64,
        )
        dtau = cp.zeros(
            shape=(total_layers, wngrid_size),
            dtype=np.float64,
        )

        intensity = cp.zeros(
            shape=(self._ngauss, wngrid_size),
            dtype=np.float64,
        )
        blackbody = cuda_blackbody(wngrid, temperature.ravel())
        tau_host = cpx.zeros_pinned(shape=(total_layers, wngrid_size), dtype=np.float64)
        if not self._fully_cuda:
            self.fallback_noncuda(layer_tau, dtau, wngrid, total_layers)

        for contrib in self.cuda_contributions:
            contrib.contribute(
                self,
                self._start_layer,
                self._end_layer,
                self._density_offset,
                0,
                density_profile,
                layer_tau,
                path_length=self._dz,
                with_layer_offset=False,
            )
            contrib.contribute(
                self,
                self._start_dtau,
                self._end_dtau,
                self._density_offset,
                0,
                density_profile,
                dtau,
                path_length=self._dz,
                with_layer_offset=False,
            )
        integral_kernal = gen_partial_kernal(self._ngauss, self.nLayers, wngrid_size)

        THREAD_PER_BLOCK_X = 64

        NUM_BLOCK_X = int(math.ceil(wngrid_size / THREAD_PER_BLOCK_X))

        integral_kernal(
            (NUM_BLOCK_X,),
            (THREAD_PER_BLOCK_X,),
            (intensity, layer_tau, dtau, blackbody),
        )
        layer_tau.get(out=tau_host)
        if keep_in_gpu:
            return (
                intensity,
                1 / self._gpu_mu_quads[:, None],
                self._gpu_wi_quads[:, None],
                tau_host,
            )

        return (
            intensity.get(),
            1 / self._mu_quads[:, None],
            self._wi_quads[:, None],
            tau_host,
        )

    def path_integral(self, wngrid, return_contrib):
        intensity, _mu, _w, tau = self.evaluate_emission(wngrid, return_contrib, keep_in_gpu=True)
        self.debug("I: %s", intensity)

        flux_total = 2.0 * np.pi * cp.sum(intensity * _w / _mu, axis=0)
        self.debug("flux_total %s", flux_total)

        return self.compute_final_flux(flux_total).ravel().get(), tau

    def fallback_noncuda(self, gpu_layer_tau, gpu_dtau, wngrid, total_layers):
        wngrid_size = wngrid.shape[0]

        dz = np.zeros(total_layers)
        dz[:-1] = np.diff(self.altitudeProfile)
        dz[-1] = self.altitudeProfile[-1] - self.altitudeProfile[-2]

        density = self.densityProfile
        layer_tau = np.zeros(shape=(total_layers, wngrid_size))
        dtau = np.zeros(shape=(total_layers, wngrid_size))
        _dtau = np.zeros(shape=(1, wngrid_size))
        _layer_tau = np.zeros(shape=(1, wngrid_size))

        # Loop upwards
        for layer in range(total_layers):
            _layer_tau[...] = 0.0
            _dtau[...] = 0.0
            for contrib in self.non_cuda_contributions:
                contrib.contribute(
                    self,
                    layer + 1,
                    total_layers,
                    0,
                    0,
                    density,
                    _layer_tau,
                    path_length=dz,
                )
                contrib.contribute(self, layer, layer + 1, 0, 0, density, _dtau, path_length=dz)

            layer_tau[layer, :] += _layer_tau[0]
            dtau[layer, :] += _dtau[0]

        gpu_layer_tau.set(layer_tau)
        gpu_dtau.set(dtau)

    def compute_final_flux(self, f_total):
        star_sed = cp.array(self._star.spectralEmissionDensity)

        self.debug("Star SED: %s", star_sed)
        # quit()
        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius
        self.debug("star_radius %s", self._star.radius)
        self.debug("planet_radius %s", self._star.radius)
        last_flux = (f_total / star_sed) * (planet_radius / star_radius) ** 2

        self.debug("last_flux %s", last_flux)

        return last_flux

    @classmethod
    def input_keywords(cls) -> tuple[str, ...]:
        return (
            "emission_cuda",
            "eclipse_cuda",
        )
