import typing as t

import cupy as cp
import numpy as np
import numpy.typing as npt
from taurex.model import OneDForwardModel

from ..opacity.cudacache import CudaCiaCache
from .cudacontrib import CudaContribution, cuda_contribute_tau


class CIACuda(CudaContribution):
    def __init__(self, cia_pairs=None):
        super().__init__("CIA")
        self._opacity_cache = CudaCiaCache()
        self._xsec_cache = {}
        self._cia_pairs = cia_pairs
        if self._cia_pairs is None:
            self._cia_pairs = []

    def build(self, model):
        super().build(model)
        self._opacity_cache.set_native_grid(model.nativeWavenumberGrid)

    def contribute(
        self,
        model,
        start_layer,
        end_layer,
        density_offset,
        layer,
        density,
        tau,
        path_length=None,
        with_layer_offset: bool = False,
    ):
        """
        Computes an integral for a single layer for the optical depth.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            A forward model

        start_layer: int
            Lowest layer limit for integration

        end_layer: int
            Upper layer limit of integration

        density_offset: int
            offset in density layer

        layer: int
            atmospheric layer being computed

        density: :obj:`array`
            density profile of atmosphere

        tau: :obj:`array`
            optical depth to store result

        path_length: :obj:`array`
            integration length

        """

        self.debug(
            " %s %s %s %s %s %s %s",
            start_layer,
            end_layer,
            density_offset,
            layer,
            density,
            tau,
            self._ngrid,
        )

        cuda_contribute_tau(
            start_layer,
            end_layer,
            density_offset,
            self.sigma_xsec,
            density * density,
            path_length,
            self._nlayers,
            self._ngrid,
            tau,
            with_layer_offset=with_layer_offset,
        )
        self.debug("DONE")

    @property
    def ciaPairs(self):
        """
        Returns list of molecular pairs involved

        Returns
        -------
        :obj:`list` of str
        """

        return self._cia_pairs

    @ciaPairs.setter
    def ciaPairs(self, value):
        self._cia_pairs = value

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

        self.debug("Preparing model with %s", wngrid.shape)
        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers
        sigma_xsec = cp.zeros(
            shape=(model.nLayers, wngrid.shape[0]),
            dtype=np.float64,
        )

        chemistry = model.chemistry
        # Loop through all active gases
        for pairName in self.ciaPairs:
            xsec = self._opacity_cache[pairName]
            cia = xsec._xsec
            cia_factor = chemistry.get_gas_mix_profile(cia.pairOne) * chemistry.get_gas_mix_profile(cia.pairTwo)

            # Get the cross section object relating to the gas

            sigma_xsec = xsec.opacity(model.temperatureProfile, cia_factor, wngrid=wngrid)

            # Temporarily assign to master cross-section
            self.sigma_xsec = sigma_xsec
            yield pairName, sigma_xsec

    def write(self, output):
        contrib = super().write(output)
        if len(self.ciaPairs) > 0:
            contrib.write_string_array("cia_pairs", self.ciaPairs)
        return contrib

    @classmethod
    def input_keywords(cls):
        return [
            "CIACuPy",
        ]
