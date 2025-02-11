import numpy as np
import numpy.typing as npt
import pytest
from taurex.cia import CIA
from taurex.util import create_grid_res


class FakeCIA(CIA):
    """Fake opacity for testing purposes."""

    def __init__(
        self,
        molecule_pair: tuple[str, str],
        num_t: int = 27,
        wn_res: int = 15000,
        wn_size: tuple[float, float] = (300, 30000),
    ):
        super().__init__("FAKE", "-".join(molecule_pair))
        self.pair = molecule_pair
        self._wavenumber_grid = create_grid_res(wn_res, *wn_size)[:, 0]
        self._temperature_grid = np.linspace(100, 10000, num_t)
        self._xsec_grid = np.random.rand(self._temperature_grid.size, self._wavenumber_grid.size)

    def find_closest_temperature_index(self, temperature: float) -> tuple[int, int]:
        """
        Finds the nearest indices for a particular temperature

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        Returns
        -------
        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
            index on temprature grid to the right of ``temperature``

        """
        from taurex.util import find_closest_pair

        t_min, t_max = find_closest_pair(self.temperatureGrid, temperature)
        return t_min, t_max

    def interp_linear_grid(self, temperature: float, t_idx_min: int, t_idx_max: int) -> float:
        """
        For a given temperature and indicies. Interpolate the cross-sections
        linearly from temperature grid to temperature ``T``

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
            index on temprature grid to the right of ``temperature``

        Returns
        -------
        out : :obj:`array`
            Interpolated cross-section

        """
        from taurex.util.math import interp_lin_only

        if temperature > self._temperature_grid.max():
            return self._xsec_grid[-1]
        elif temperature < self._temperature_grid.min():
            return self._xsec_grid[0]

        temp_max = self._temperature_grid[t_idx_max]
        temp_min = self._temperature_grid[t_idx_min]
        fx0 = self._xsec_grid[t_idx_min]
        fx1 = self._xsec_grid[t_idx_max]

        return interp_lin_only(fx0, fx1, temperature, temp_min, temp_max)

    def compute_cia(self, temperature: float) -> npt.NDArray[np.float64]:
        """
        Computes the collisionally induced absorption cross-section
        using our native temperature and cross-section grids

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        out : :obj:`array`
            Temperature interpolated cross-section

        """
        indicies = self.find_closest_temperature_index(temperature)
        return self.interp_linear_grid(temperature, *indicies)

    @property
    def moleculeName(self) -> str:
        """Name of molecule."""
        return self._molecule_name

    @property
    def xsecGrid(self) -> npt.NDArray[np.float64]:
        """Opacity grid."""
        return self._xsec_grid

    @property
    def wavenumberGrid(self) -> npt.NDArray[np.float64]:
        """Wavenumber grid."""
        return self._wavenumber_grid

    @property
    def temperatureGrid(self) -> npt.NDArray[np.float64]:
        """Temperature grid."""
        return self._temperature_grid


@pytest.fixture
def cia_opac():
    fo = FakeCIA(("H2", "H2"), wn_res=4)
    from taurex.cache import CIACache

    oc = CIACache()
    oc.cia_dict.clear()
    oc.add_cia(fo)
    from taurex_cupy.cia.cudacia import CudaCIA

    co = CudaCIA("H2-H2")
    yield fo, co
    oc.cia_dict.clear()
    del co
    del fo


@pytest.mark.parametrize("temperature", np.linspace(100, 10000, 10))
def test_cia_opacity(cia_opac, temperature):
    """Test CUDA opacity to see if it matches the CPU version."""
    fo, co = cia_opac
    from taurex_cupy.cia.cudacia import CudaCIA

    co = CudaCIA("H2-H2")
    cpu_result = fo.cia(temperature=temperature)
    gpu_result = co.opacity(temperature=temperature, mix=1.0)

    gpu_return = gpu_result.get()

    np.testing.assert_almost_equal(cpu_result, gpu_return[0])
