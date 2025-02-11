"""Direct imaging model."""

import numpy as np
import numpy.typing as npt
from taurex.model.directimage import compute_direct_image_final_flux

from .eclipse import EmissionCudaModel


class DirectImageModel(EmissionCudaModel):
    """A forward model for direct imaging of exo-planets."""

    def compute_final_flux(self, f_total: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the final flux.

        This is the emission flux that is observed at the telescope directly
        from an exo-planet.

        """
        return compute_direct_image_final_flux(f_total, self._planet.fullRadius, self._star.distance * 3.08567758e16)

    @classmethod
    def input_keywords(cls) -> tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "direct_cuda",
            "directimage_cuda",
        )
