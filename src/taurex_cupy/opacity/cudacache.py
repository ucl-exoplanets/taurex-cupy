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
    """Cache for automated CPU->GPU transfer of opacity objects."""

    def init(self) -> None:
        """Initialisation."""
        self.opacity_dict: dict[str, T] = {}
        self.log = Logger(self.__class__.__name__)
        self._wngrid = None

    def set_native_grid(self, native_grid: npt.NDArray[np.floating]) -> None:
        """All opacities are homogenised to a wavenumber grid.

        All opacities are homogenised to a wavenumber grid using interpolation.

        Args:
            native_grid: Native wavenumber grid


        """
        if self._wngrid is None or not np.array_equal(native_grid, self._wngrid):
            self.log.info("Re-homogenizing native grids!")
            self._wngrid = native_grid

            for opac in self.opacity_dict.values():
                opac.transfer_xsec_grid(self._wngrid)

    def create_object(self, key: str, wngrid: npt.NDArray[np.floating]) -> T:
        """Method to create the opacity object.

        Args:
            key: molecule name
            wngrid: wavenumber grid

        Returns:
            Created opacity object
        """

        raise NotImplementedError

    def __getitem__(self, key: str) -> T:
        """Get item method for the cache.

        Args:
            key: molecule name

        Returns:
            Opacity object

        Raises:
            KeyError: If the key is not found

        """
        if key in self.opacity_dict:
            return self.opacity_dict[key]
        else:
            # Try a load of the opacity
            self.opacity_dict[key] = self.create_object(key, wngrid=self._wngrid)
            return self.opacity_dict[key]

    def clear_cache(self) -> None:
        """Clears all currently loaded cross-sections.

        Useful in freeing up memory when the cache is no longer needed.

        """
        self.opacity_dict.clear()


class CudaOpacityCache(CudaCache[CudaOpacity]):
    """Cache for automated CPU->GPU transfer of absorption cross-section objects."""

    def create_object(self, key: str, wngrid: npt.NDArray[np.floating]) -> CudaOpacity:
        """Create the absorption cross-section object.

        Args:
            key: molecule name
            wngrid: wavenumber grid

        Returns:
            Created absorption cross-section object

        """
        return CudaOpacity(key, wngrid=self._wngrid)


class CudaCiaCache(CudaCache[CudaCIA]):
    """Cache for automated CPU->GPU transfer of collisionally induced absorption objects."""

    def create_object(self, key: str, wngrid: npt.NDArray[np.floating]) -> CudaCIA:
        """Create the collisionally induced absorption object.

        Args:
            key: molecule name
            wngrid: wavenumber grid

        Returns:
            Created collisionally induced absorption object

        """
        return CudaCIA(key, wngrid=self._wngrid)
