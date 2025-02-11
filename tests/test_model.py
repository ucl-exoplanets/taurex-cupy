"""Test models."""

import numpy as np
import pytest
from taurex.contributions import (
    FlatMieContribution,
    LeeMieContribution,
    RayleighContribution,
    SimpleCloudsContribution,
)

from taurex_cupy.contributions import (
    LeeMieCuda,
    RayleighCuda,
    SimpleCloudsCuda,
)
from taurex_cupy.util import FakeCIA

WNRES = 100


@pytest.fixture
def opac():
    from taurex.opacity.fakeopacity import FakeOpacity

    fo = FakeOpacity("H2O", wn_res=WNRES)
    fc = FakeCIA(("H2", "H2"), wn_res=WNRES)
    from taurex.cache import CIACache, OpacityCache

    oc = OpacityCache()
    cc = CIACache()
    oc.clear_cache()
    cc.cia_dict.clear()
    oc.add_opacity(fo)
    cc.add_cia(fc)
    from taurex_cupy.cia.cudacia import CudaCIA
    from taurex_cupy.opacity.cudaopacity import CudaOpacity

    co = CudaOpacity("H2O")
    cc_cuda = CudaCIA("H2-H2")
    yield fo, co, cc_cuda
    oc.clear_cache()
    cc.cia_dict.clear()
    del co
    del fo
    del fc
    del cc_cuda


def test_transit(opac):
    from taurex.contributions import AbsorptionContribution
    from taurex.model import TransmissionModel

    from taurex_cupy.contributions.absorption import AbsorptionCuda
    from taurex_cupy.model.transit import TransmissionCudaModel

    tm_cpu = TransmissionModel()
    tm_gpu = TransmissionCudaModel()
    tm_cpu.add_contribution(AbsorptionContribution())
    tm_gpu.add_contribution(AbsorptionCuda())

    tm_cpu.build()
    tm_gpu.build()

    res = tm_cpu.model()
    res_gpu = tm_gpu.model()

    wngrid, flux, tau, extra = res
    wngrid_gpu, flux_gpu, tau_gpu, extra_gpu = res_gpu

    np.testing.assert_allclose(wngrid, wngrid_gpu)
    np.testing.assert_allclose(tau, tau_gpu)
    np.testing.assert_allclose(flux, flux_gpu)


def test_transit_non_cuda(opac):
    from taurex.contributions import AbsorptionContribution
    from taurex.model import TransmissionModel

    from taurex_cupy.model.transit import TransmissionCudaModel

    tm_cpu = TransmissionModel()
    tm_gpu = TransmissionCudaModel()
    tm_cpu.add_contribution(AbsorptionContribution())
    tm_gpu.add_contribution(AbsorptionContribution())

    tm_cpu.build()
    tm_gpu.build()

    res = tm_cpu.model()
    res_gpu = tm_gpu.model()

    wngrid, flux, tau, extra = res
    wngrid_gpu, flux_gpu, tau_gpu, extra_gpu = res_gpu

    np.testing.assert_allclose(wngrid, wngrid_gpu)
    np.testing.assert_allclose(tau, tau_gpu)
    np.testing.assert_allclose(flux, flux_gpu)


def test_cia(opac):
    from taurex.contributions import AbsorptionContribution, CIAContribution
    from taurex.model import TransmissionModel

    from taurex_cupy.contributions.absorption import AbsorptionCuda
    from taurex_cupy.contributions.cia import CIACuda
    from taurex_cupy.model.transit import TransmissionCudaModel

    tm_cpu = TransmissionModel()
    tm_gpu = TransmissionCudaModel()
    tm_cpu.add_contribution(AbsorptionContribution())
    tm_cpu.add_contribution(CIAContribution())
    tm_gpu.add_contribution(AbsorptionCuda())
    tm_gpu.add_contribution(CIACuda())

    tm_cpu.build()
    tm_gpu.build()

    res = tm_cpu.model()
    res_gpu = tm_gpu.model()

    wngrid, flux, tau, extra = res
    wngrid_gpu, flux_gpu, tau_gpu, extra_gpu = res_gpu

    np.testing.assert_allclose(wngrid, wngrid_gpu)
    np.testing.assert_allclose(tau, tau_gpu)
    # Check its not zero)
    np.testing.assert_allclose(flux, flux_gpu)


@pytest.mark.parametrize(
    "contrib",
    [
        (RayleighContribution, RayleighCuda),
        (SimpleCloudsContribution, SimpleCloudsCuda),
        (LeeMieContribution, LeeMieCuda),
    ],
)
def test_various_contributions(opac, contrib):
    from taurex.contributions import AbsorptionContribution
    from taurex.model import TransmissionModel

    from taurex_cupy.contributions.absorption import AbsorptionCuda
    from taurex_cupy.model.transit import TransmissionCudaModel

    cpu_klass, gpu_klass = contrib

    tm_cpu = TransmissionModel()
    tm_gpu = TransmissionCudaModel()

    tm_cpu.add_contribution(AbsorptionContribution())
    tm_cpu.add_contribution(cpu_klass())
    tm_gpu.add_contribution(AbsorptionCuda())
    tm_gpu.add_contribution(gpu_klass())

    tm_cpu.build()
    tm_gpu.build()

    res = tm_cpu.model()
    res_gpu = tm_gpu.model()

    wngrid, flux, tau, extra = res
    wngrid_gpu, flux_gpu, tau_gpu, extra_gpu = res_gpu

    np.testing.assert_allclose(wngrid, wngrid_gpu)
    np.testing.assert_allclose(tau, tau_gpu)
    np.testing.assert_allclose(flux, flux_gpu)


# Something is wrong with FlatMie, needs a check.
@pytest.mark.skip
def test_flatmie_contribution(opac):
    from taurex.contributions import AbsorptionContribution
    from taurex.model import TransmissionModel

    from taurex_cupy.contributions.absorption import AbsorptionCuda
    from taurex_cupy.contributions.flatmie import FlatMieCuda
    from taurex_cupy.model.transit import TransmissionCudaModel

    tm_cpu = TransmissionModel()
    tm_gpu = TransmissionCudaModel()

    tm_cpu.add_contribution(AbsorptionContribution())
    tm_cpu.add_contribution(FlatMieContribution())
    tm_gpu.add_contribution(AbsorptionCuda())
    tm_gpu.add_contribution(FlatMieCuda())

    tm_cpu.build()
    tm_gpu.build()

    res = tm_cpu.model()
    res_gpu = tm_gpu.model()

    wngrid, flux, tau, extra = res
    wngrid_gpu, flux_gpu, tau_gpu, extra_gpu = res_gpu

    np.testing.assert_allclose(wngrid, wngrid_gpu)
    np.testing.assert_allclose(tau, tau_gpu)
    np.testing.assert_allclose(flux, flux_gpu)


def test_eclipse(opac):
    from taurex.contributions import AbsorptionContribution
    from taurex.model import EmissionModel
    from taurex.temperature import Guillot2010

    from taurex_cupy.contributions.absorption import AbsorptionCuda
    from taurex_cupy.model.eclipse import EmissionCudaModel

    tm_cpu = EmissionModel(temperature_profile=Guillot2010())
    tm_gpu = EmissionCudaModel(temperature_profile=Guillot2010())
    tm_cpu.add_contribution(AbsorptionContribution())
    tm_gpu.add_contribution(AbsorptionCuda())

    tm_cpu.build()
    tm_gpu.build()

    res = tm_cpu.model()
    res_gpu = tm_gpu.model()

    wngrid, flux, tau, extra = res
    wngrid_gpu, flux_gpu, tau_gpu, extra_gpu = res_gpu
    np.testing.assert_equal(tm_cpu._mu_quads, tm_gpu._mu_quads)
    np.testing.assert_equal(tm_cpu._wi_quads, tm_gpu._wi_quads)
    np.testing.assert_allclose(wngrid, wngrid_gpu)
    np.testing.assert_allclose(tau, tau_gpu)
    np.testing.assert_allclose(flux, flux_gpu, rtol=1e-5)


def test_eclipse_non_cuda(opac):
    from taurex.contributions import AbsorptionContribution
    from taurex.model import EmissionModel

    from taurex_cupy.model.eclipse import EmissionCudaModel

    tm_cpu = EmissionModel()
    tm_gpu = EmissionCudaModel()
    tm_cpu.add_contribution(AbsorptionContribution())
    tm_gpu.add_contribution(AbsorptionContribution())

    tm_cpu.build()
    tm_gpu.build()

    res = tm_cpu.model()
    res_gpu = tm_gpu.model()

    wngrid, flux, tau, extra = res
    wngrid_gpu, flux_gpu, tau_gpu, extra_gpu = res_gpu

    np.testing.assert_allclose(wngrid, wngrid_gpu)
    np.testing.assert_allclose(tau, tau_gpu)
    np.testing.assert_allclose(flux, flux_gpu, rtol=1e-5)
