import math
import typing as t
from functools import lru_cache

import cupy as cp
import numpy as np
from taurex.contributions import Contribution
from taurex.model import OneDForwardModel


@lru_cache(maxsize=400)
def _contribute_tau_kernel_II(
    nlayers: int, grid_size: int, with_layer_offset: bool = True, start_layer: int = 0
) -> cp.RawKernel:
    extra = "+layer" if with_layer_offset else ""

    code = f"""
    extern "C" __global__ void contribute_tau(double* dest, const double* __restrict__ sigma,
                                   const double* __restrict__ density, const double* __restrict__ path,
                                   const int* __restrict__ startK, const int* __restrict__ endK,
                                   const int* __restrict__ density_offset, const int total_layers)
    {{
        __shared__ double density_cache[{nlayers}];
        const unsigned int grid = (blockIdx.x * blockDim.x) + threadIdx.x;
        const unsigned int layer = (blockIdx.y * blockDim.y) + threadIdx.y + {start_layer};

        //Cache path and density in shared memory
        const int total_threads=blockDim.x*blockDim.y;
        const int local_thread_id = threadIdx.x + threadIdx.y*blockDim.x;

        for(int i = local_thread_id; i < {nlayers}; i+=total_threads){{
            density_cache[i] = density[i];
        }}
        __syncthreads();


        if ( grid >= {grid_size} )
            return;
        if (layer >= {nlayers})
            return;

        const unsigned int _startK = startK[layer];
        const unsigned int _endK = endK[layer];
        const unsigned int _offset = density_offset[layer];
        double _result = 0.0;
        for (unsigned int k = _startK; k < _endK; k++)
        {{
            double _path = path[layer * {nlayers} + k];
            double _density = density_cache[k+_offset];
            _result += sigma[(k{extra})*{grid_size} + grid]*_path*_density;
        }}
        dest[layer*{grid_size} + grid] += _result;


    }}

    """

    kernel = cp.RawKernel(code, "contribute_tau")
    return kernel


def cuda_contribute_tau(
    startK: cp.ndarray,
    endK: cp.ndarray,
    density_offset: cp.ndarray,
    sigma: cp.ndarray,
    density: cp.ndarray,
    path: cp.ndarray,
    nlayers: int,
    ngrid: int,
    tau: cp.ndarray | None = None,
    with_layer_offset: bool = True,
    start_layer: int = 0,
    total_layers: t.Optional[int] = None,
):
    kernel = _contribute_tau_kernel_II(nlayers, ngrid, with_layer_offset=with_layer_offset, start_layer=start_layer)
    my_tau = tau
    if total_layers is None:
        total_layers = nlayers
    if my_tau is None:
        my_tau = cp.zeros(shape=(nlayers, ngrid), dtype=np.float64)

    THREAD_PER_BLOCK_X = 16
    THREAD_PER_BLOCK_Y = 16

    # THREAD_PER_BLOCK_X = 128
    # THREAD_PER_BLOCK_Y = 1
    NUM_BLOCK_Y = int(math.ceil((total_layers) / THREAD_PER_BLOCK_Y))
    NUM_BLOCK_X = int(math.ceil((ngrid) / THREAD_PER_BLOCK_X))
    # NUM_BLOCK_Y = 1

    kernel(
        (NUM_BLOCK_X, NUM_BLOCK_Y, 1),
        (THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, 1),
        (
            my_tau,
            sigma,
            density,
            path,
            startK,
            endK,
            density_offset,
            np.int32(total_layers),
        ),
    )
    if tau is None:
        return my_tau


class CudaContribution(Contribution):
    def __init__(self, name):
        super().__init__(name)
        self._is_cuda_model = False

    def contribute(
        self,
        model: OneDForwardModel,
        start_layer: cp.ndarray,
        end_layer: cp.ndarray,
        density_offset: cp.ndarray,
        layer: int,
        density: cp.ndarray,
        tau: cp.ndarray,
        path_length: t.Optional[cp.ndarray] = None,
        with_layer_offset: bool = True,
    ):
        """
        Computes an integral for a single layer for the optical depth.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            A forward model

        start_layer: int
            Lowest layer limit for integration

        end_layer: int
            Upper layer limit of integration

        density_offset: int
            offset in density layer

        layer: int
            atmospheric layer being computed

        density: :obj:`array`
            density profile of atmosphere

        tau: :obj:`array`
            optical depth to store result

        path_length: :obj:`array`
            integration length

        """
        self.debug(
            " %s %s %s %s %s %s %s",
            start_layer,
            end_layer,
            density_offset,
            layer,
            density,
            tau,
            self._ngrid,
        )

        cuda_contribute_tau(
            start_layer,
            end_layer,
            density_offset,
            self.sigma_xsec,
            density,
            path_length,
            self._nlayers,
            self._ngrid,
            tau,
            with_layer_offset=with_layer_offset,
        )

        self.debug("DONE")

    def prepare(self, model, wngrid):
        """

        Used to prepare the contribution for the calculation.
        Called before the forward model performs the main optical depth
        calculation. Default behaviour is to loop through :func:`prepare_each`
        and sum all results into a single cross-section.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid
        """
        del self.sigma_xsec
        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers

        sigma_xsec = cp.zeros(shape=(self._nlayers, self._ngrid), dtype=np.float64)

        for gas, sigma in self.prepare_each(model, wngrid):
            self.debug("Gas %s", gas)
            self.debug("Sigma %s", sigma)
            sigma_xsec += sigma

        self.sigma_xsec = sigma_xsec
        self.debug("Final sigma is %s", self.sigma_xsec)
        self.info("Done")
