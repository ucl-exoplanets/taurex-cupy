import numpy as np
import pytest

from taurex_cupy.util import FakeCIA


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
