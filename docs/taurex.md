# TauREx Overview

The TauREx-CuPy plugin supplies drop-in models and radiative transfer functions
that allow for the exploitation of nVidia CUDA hardware.

TauREx-CuPy should be compatible with the majority of plugins available. However, some
that add new _radiative models_ will not work unless they are specifically updated to be compatible
with CuPy

!!! warning
TauREx-CUDA is only compatible with the cross-section opacity method.

To make use of the plugin, you can simply replace existing models and contributions
with the CUDA-enabled ones.

## TauREx Input File

For using the `taurex` program, exploiting CUDA requires replacing keywords with `cuda/CUDA` variants.
For example if we have a transit spectrum input file:

```
[Model]
model_type = transmission
    [[Absorption]]

    [[CIA]]
    cia_pairs = H2-H2, H2-He

    [[Rayleigh]]
```

The CUDA enabled version is simply:

```
[Model]
model_type = transmission_cuda
    [[AbsorptionCuda]]

    [[CIACuda]]
    cia_pairs = H2-H2, H2-He

    [[RayleighCuda]]
```

The CUDA forward models can function with non-CUDA enabled contribution functions.
We can mix the C++ Mie scattering code for the eclipse version:

```
[Model]
model_type = emission_cuda
    [[AbsorptionCuda]]

    [[CIACuda]]
    cia_pairs = H2-H2, H2-He

    [[RayleighCuda]]

    [[BHMie]]
```

When non-CUDA contributions are included, they are first computed on the CPU before their
results are transferred and the model is continued on the GPU.

!!! warning
The oppisite is not true. CUDA contributions are only compatible with CUDA forward models.
Attempting to use them in non-CUDA forward models will result in an error.

The available replacements are:

| taurex3        | taurex-cupy         |
| -------------- | ------------------- |
| `transmission` | `transmission_cuda` |
| `emission`     | `emission_cuda`     |
| `Absorption`   | `AbsorptionCuda`    |
| `CIA`          | `CIACuda`           |
| `Rayleigh`     | `RayleighCuda`      |
| `SimpleClouds` | `SimpleCloudsCuda`  |
| `LeeMie`       | `LeeMieCuda`        |

## TauREx via script or notebook

To make use of the CUDA models in a script or notebook, you can simply import the CUDA models
from the top level `taurex_cupy` module.

```python
from taurex_cupy import TransmissionCudaModel
from taurex_cupy import AbsorptionCuda, CIACuda, RayleighCuda
from taurex.chemistry import TaurexChemistry, ConstantGas

chemistry = TaurexChemistry(
    fill_gases=["H2","He"],
    ratio=0.17567,
)
chemistry.addGas(ConstantGas("H2O", 1e-3)).addGas(ConstantGas("CH4", 1e-4))

tm_gpu = TransmissionCudaModel(chemistry=chemistry)
tm_gpu.add_contribution(AbsorptionCuda())
tm_gpu.add_contribution(CIACuda(cia_pairs=["H2-H2"]))
tm_gpu.add_contribution(RayleighCuda())

tm_gpu.build()

wngrid, rprs, tau, _ = tm_gpu.model()
```

## Interoperability

The CUDA models are designed to be compatible with the majority of TauREx plugins.
For example, using the `acepython` plugin for the chemistry:

```bash
pip install acepython
```

Then running the following script:

```python
from taurex_cupy import TransmissionCudaModel
from taurex_cupy import AbsorptionCuda, CIACuda, RayleighCuda
from acepython.taurex import ACEChemistry

chemistry = ACEChemistry()
tm_gpu = TransmissionCudaModel(chemistry=chemistry)
tm_gpu.add_contribution(AbsorptionCuda())
tm_gpu.add_contribution(CIACuda(cia_pairs=["H2-H2"]))
tm_gpu.add_contribution(RayleighCuda())

tm_gpu.build()

wngrid, rprs, tau, _ = tm_gpu.model()
```

Or replacing the plugin in the input file:

```bash
[Chemistry]
chemistry = ace

[Model]
model_type = transmission_cuda
    [[AbsorptionCuda]]

    [[CIACuda]]
    cia_pairs = H2-H2, H2-He

    [[RayleighCuda]]
```
