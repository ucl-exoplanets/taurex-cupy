"""Mie scattering using Lee et al. 2013 formalism."""

import typing as t

import cupy as cp
import numpy as np
import numpy.typing as npt
from taurex.contributions.leemie import LeeMieContribution
from taurex.data.fittable import fitparam
from taurex.model import OneDForwardModel
from taurex.output import OutputGroup

from .cudacontrib import CudaContribution


class LeeMieCuda(CudaContribution):
    """Computes Mie scattering contribution to optica depth.

    Formalism taken from: Lee et al. 2013, ApJ, 778, 97

    Parameters
    ----------

    lee_mie_radius: float
        Particle radius in um

    lee_mie_q: float
        Extinction coefficient

    lee_mie_mix_ratio: float
        Mixing ratio in atmosphere

    lee_mie_bottomP: float
        Bottom of cloud deck in Pa

    lee_mie_topP: float
        Top of cloud deck in Pa


    """

    def __init__(
        self,
        lee_mie_radius: t.Optional[float] = 0.01,
        lee_mie_q: t.Optional[float] = 40,
        lee_mie_mix_ratio: t.Optional[float] = 1e-10,
        lee_mie_bottomP: t.Optional[float] = -1,
        lee_mie_topP: t.Optional[float] = -1,
    ) -> None:
        super().__init__("Mie")

        self._mie_radius = lee_mie_radius
        self._mie_q = lee_mie_q
        self._mie_mix = lee_mie_mix_ratio
        self._mie_bottom_pressure = lee_mie_bottomP
        self._mie_top_pressure = lee_mie_topP

    @fitparam(
        param_name="lee_mie_radius",
        param_latex=r"$R^{lee}_{\mathrm{mie}}$",
        default_fit=False,
        default_bounds=[0.01, 0.5],
    )
    def mieRadius(self) -> float:
        """Particle radius in um."""
        return self._mie_radius

    @mieRadius.setter
    def mieRadius(self, value: float) -> None:
        """Particle radius in um."""
        self._mie_radius = value

    @fitparam(
        param_name="lee_mie_q",
        param_latex=r"$Q_\mathrm{ext}$",
        default_fit=False,
        default_bounds=[-10, 1],
    )
    def mieQ(self) -> float:
        """Extinction coefficient."""
        return self._mie_q

    @mieQ.setter
    def mieQ(self, value: float) -> None:
        self._mie_q = value

    @fitparam(
        param_name="lee_mie_topP",
        param_latex=r"$P^{lee}_\mathrm{top}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[-1, 1],
    )
    def mieTopPressure(self) -> float:
        """Pressure at top of cloud deck in Pa."""
        return self._mie_top_pressure

    @mieTopPressure.setter
    def mieTopPressure(self, value: float) -> None:
        """Pressure at top of cloud deck in Pa."""
        self._mie_top_pressure = value

    @fitparam(
        param_name="lee_mie_bottomP",
        param_latex=r"$P^{lee}_\mathrm{bottom}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[-1, 1],
    )
    def mieBottomPressure(self) -> float:
        """Pressure at bottom of cloud deck in Pa."""
        return self._mie_bottom_pressure

    @mieBottomPressure.setter
    def mieBottomPressure(self, value: float) -> None:
        """Pressure at bottom of cloud deck in Pa."""
        self._mie_bottom_pressure = value

    @fitparam(
        param_name="lee_mie_mix_ratio",
        param_latex=r"$\chi^{lee}_\mathrm{mie}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[-1, 1],
    )
    def mieMixing(self) -> float:
        """Mixing ratio in atmosphere."""
        return self._mie_mix

    @mieMixing.setter
    def mieMixing(self, value: float) -> None:
        """Mixing ratio in atmosphere."""
        self._mie_mix = value

    def prepare_each(
        self, model: OneDForwardModel, wngrid: npt.NDArray[np.float64]
    ) -> t.Generator[tuple[str, npt.NDArray[np.float64]], None, None]:
        """Compute and weights the mie opacity for the pressure regions given.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Lee`` and the weighted mie opacity.

        """
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        pressure_profile = cp.asarray(model.pressureProfile)

        bottom_pressure = self.mieBottomPressure
        if bottom_pressure < 0:
            bottom_pressure = model.pressureProfile[0]

        top_pressure = self.mieTopPressure
        if top_pressure < 0:
            top_pressure = model.pressureProfile[-1]

        wltmp = cp.asarray(10000 / wngrid)

        a = self.mieRadius

        x = 2.0 * np.pi * a / wltmp
        self.debug("wngrid %s", wngrid)
        self.debug("x %s", x)
        q_ext = 5.0 / (self.mieQ * x ** (-4.0) + x ** (0.2))

        sigma_xsec = cp.zeros(shape=(self._nlayers, wngrid.shape[0]))

        # This must transform um to the xsec format in TauREx (m2)
        am = a * 1e-6

        sigma_mie = q_ext * np.pi * (am**2.0)

        self.debug("q_ext %s", q_ext)
        self.debug("radius um %s", a)
        self.debug("sigma %s", sigma_mie)

        self.debug("bottome_pressure %s", bottom_pressure)
        self.debug("top_pressure %s", top_pressure)

        cloud_filter = (pressure_profile <= bottom_pressure) & (pressure_profile >= top_pressure)

        sigma_xsec[cloud_filter, ...] = sigma_mie * self.mieMixing

        self.sigma_xsec = sigma_xsec

        self.debug("final xsec %s", self.sigma_xsec)

        yield "Lee", sigma_xsec

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write output group."""
        contrib = super().write(output)
        contrib.write_scalar("lee_mie_radius", self._mie_radius)
        contrib.write_scalar("lee_mie_q", self._mie_q)
        contrib.write_scalar("lee_mie_mix_ratio", self._mie_mix)
        contrib.write_scalar("lee_mie_bottomP", self._mie_bottom_pressure)
        contrib.write_scalar("lee_mie_topP", self._mie_top_pressure)
        return contrib

    @classmethod
    def input_keywords(cls) -> tuple[str]:
        """Input keywords.""" ""
        return ("LeeMieCuda",)

    BIBTEX_ENTRIES = LeeMieContribution.BIBTEX_ENTRIES
