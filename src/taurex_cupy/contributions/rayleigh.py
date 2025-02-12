"""Rayleigh contribution to optical depth."""

import typing as t

import cupy as cp
import numpy as np
import numpy.typing as npt
from taurex.contributions.rayleigh import RayleighContribution
from taurex.model import OneDForwardModel

from .cudacontrib import CudaContribution


class RayleighCuda(CudaContribution):
    """
    Computes the contribution to the optical depth
    occuring from molecular Rayleigh.
    """

    def __init__(self):
        super().__init__("Rayleigh")
        self.sigmas = {}
        self._current_grid = None

    def build(self, model: OneDForwardModel):
        super().build(model)
        self._mix_array = cp.zeros(shape=(model.nLayers,), dtype=np.float64)
        self._current_grid = None
        self.sigmas.clear()

    def prepare_each(
        self, model: OneDForwardModel, wngrid: npt.NDArray[np.floating]
    ) -> t.Iterator[tuple[str, npt.NDArray[np.floating]]]:
        """
        Prepares each molecular opacity by weighting them
        by their mixing ratio in the atmosphere

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            Name of molecule and weighted opacity

        """
        from taurex.util.scattering import rayleigh_sigma_from_name

        self.debug("Preparing model with %s", wngrid.shape)
        self._ngrid = wngrid.shape[0]
        molecules = model.chemistry.activeGases + model.chemistry.inactiveGases
        if self._current_grid is None or not np.array_equal(self._current_grid, wngrid):
            gpu_wngrid = cp.asarray(wngrid, dtype=np.float64)
            for gasname in molecules:
                self.sigmas[gasname] = rayleigh_sigma_from_name(gasname, gpu_wngrid)

        for gasname in molecules:
            if np.max(model.chemistry.get_gas_mix_profile(gasname)) == 0.0:
                continue
            sigma = self.sigmas.get(gasname, None)

            if sigma is not None:
                final_sigma = sigma[None, :] * cp.array(model.chemistry.get_gas_mix_profile(gasname)[:, None])
                self.sigma_xsec = final_sigma
                yield gasname, final_sigma

    @classmethod
    def input_keywords(cls):
        return [
            "RayleighCuda",
        ]

    BIBTEX_ENTRIES = RayleighContribution.BIBTEX_ENTRIES
