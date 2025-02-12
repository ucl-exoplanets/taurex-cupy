# Installation

TauREx-CuPy makes use of the [CuPy](https://cupy.dev/) library. While you can install `taurex-cupy` directly from PyPi, this will
require compiling CuPy which can take a long time. It is therefore recommended that you install CuPy beforehand
using the methods outlined in the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

`taurex-cupy` is installed using the following command:

```bash
pip install taurex-cupy
```

This will install the latest version of `taurex-cupy` from PyPi. If you want to install the latest development version
you can install it directly from the GitHub repository:

```bash
pip install git+https://https://github.com/ucl-exoplanets/taurex-cupy.git
```

You can check the installation by running the following command:

```bash
taurex --plugins
```

You should see it available as a successfully loaded plugin:

```
Successfully loaded plugins
---------------------------
cupy


Failed plugins
---------------------------
```
