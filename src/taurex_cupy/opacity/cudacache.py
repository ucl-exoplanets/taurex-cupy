"""Opacity cache for CUDA"""

import typing as t

import numpy as np
import numpy.typing as npt
from taurex.cache.singleton import Singleton
from taurex.log import Logger

from ..cia.cudacia import CudaCIA
from .cudaopacity import CudaOpacity

T = t.TypeVar("T", bound=CudaOpacity | CudaCIA)


class CudaCache(t.Generic[T], Singleton):
    def init(self) -> None:
        self.opacity_dict: dict[str, T] = {}
        self.log = Logger(self.__class__.__name__)
        self._wngrid = None

    def set_native_grid(self, native_grid: npt.NDArray[np.floating]) -> None:
        if self._wngrid is None or not np.array_equal(native_grid, self._wngrid):
            self.log.info("Re-homogenizing native grids!")
            self._wngrid = native_grid

            for opac in self.opacity_dict.values():
                opac.transfer_xsec_grid(self._wngrid)

    def create_object(self, key: str, wngrid: npt.NDArray[np.floating]) -> T:
        raise NotImplementedError

    def __getitem__(self, key: str) -> T:
        """
        For a molecule return the relevant :class:`~taurex.opacity.opacity.Opacity` object.


        Parameter
        ---------
        key : str
            molecule name

        Returns
        -------
        :class:`~taurex.opacity.pickleopacity.PickleOpacity`
            Cross-section object desired

        Raise
        -----
        Exception
            If molecule could not be loaded/found

        """
        if key in self.opacity_dict:
            return self.opacity_dict[key]
        else:
            # Try a load of the opacity
            self.opacity_dict[key] = self.create_object(key, wngrid=self._wngrid)
            return self.opacity_dict[key]

    def clear_cache(self) -> None:
        """
        Clears all currently loaded cross-sections
        """
        self.opacity_dict = {}


class CudaOpacityCache(CudaCache[CudaOpacity]):
    def create_object(self, key: str, wngrid: npt.NDArray[np.floating]) -> CudaOpacity:
        return CudaOpacity(key, wngrid=self._wngrid)


class CudaCiaCache(CudaCache[CudaCIA]):
    def create_object(self, key: str, wngrid: npt.NDArray[np.floating]) -> CudaCIA:
        return CudaCIA(key, wngrid=self._wngrid)
