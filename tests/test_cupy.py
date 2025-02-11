"""Test the plugin itself"""

import pytest

from taurex_cupy import (
    AbsorptionCuda,
    CIACuda,
    DirectImageCudaModel,
    EmissionCudaModel,
    TransmissionCudaModel,
)


@pytest.fixture(scope="session")
def class_factory():
    from taurex.parameter.classfactory import ClassFactory

    cf = ClassFactory()

    yield cf


@pytest.mark.parametrize(
    "test_case",
    [
        ("transit_cuda", TransmissionCudaModel),
        ("eclipse_cuda", EmissionCudaModel),
        ("transmission_cuda", TransmissionCudaModel),
        ("emission_cuda", EmissionCudaModel),
        ("directimage_cuda", DirectImageCudaModel),
        ("AbsorptionCuda", AbsorptionCuda),
        ("CIACuda", CIACuda),
    ],
)
def test_detected_by_taurex(class_factory, test_case):
    cf = class_factory

    keyword, klass = test_case

    assert cf.find_klass_from_keyword(keyword) == klass
