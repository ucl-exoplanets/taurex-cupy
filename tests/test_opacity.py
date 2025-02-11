import numpy as np
import pytest

NLAYERS = 100


@pytest.fixture
def opac():
    from taurex.opacity.fakeopacity import FakeOpacity

    fo = FakeOpacity("H2O", wn_res=4)
    from taurex.cache import OpacityCache

    oc = OpacityCache()
    oc.clear_cache()
    oc.add_opacity(fo)
    from taurex_cupy.opacity.cudaopacity import CudaOpacity

    co = CudaOpacity("H2O")
    yield fo, co
    oc.clear_cache()
    del co
    del fo


def test_cuda_opacity_same_data(opac):
    fo, co = opac

    np.testing.assert_equal(fo.xsecGrid, co._gpu_grid.get())
    np.testing.assert_equal(fo.wavenumberGrid, co._wngrid)
    np.testing.assert_equal(fo.temperatureGrid, co._gpu_tgrid.get())
    np.testing.assert_equal(fo.logPressure, co._gpu_pgrid.get())


def test_cuda_opacity(opac):
    """Test CUDA opacity to see if it matches the CPU version."""
    fo, co = opac
    from taurex_cupy.opacity.cudaopacity import CudaOpacity

    co = CudaOpacity("H2O")
    for temperature in np.linspace(100, 10000, 10):
        for pressure in np.logspace(-6, 6, 10):
            cpu_result = fo.opacity(temperature=temperature, pressure=pressure)
            gpu_result = co.opacity(temperature=temperature, pressure=pressure, mix=1.0)

            gpu_return = gpu_result.get()

            np.testing.assert_almost_equal(cpu_result, gpu_return[0])


def test_cuda_opacity_batch(opac):
    """Test CUDA opacity to see if it matches the CPU version."""
    fo, co = opac
    from taurex_cupy.opacity.cudaopacity import CudaOpacity

    co = CudaOpacity("H2O")

    temperatures = np.linspace(100, 10000, 10)
    pressures = np.logspace(-6, 6, 10)
    mix = np.random.rand(10)
    cpu_result = np.array([fo.opacity(temperature=t, pressure=p) * m for t, p, m in zip(temperatures, pressures, mix)])

    gpu_result = co.opacity(temperatures, pressures, mix)

    gpu_return = gpu_result.get()

    np.testing.assert_almost_equal(cpu_result, gpu_return)
