# PyFixest on the GPU via CuPy

Besides JAX, it is possible to run PyFixest on the GPU via CuPy (linux and windows). Instead of applying the alternating projections algorithm to demean fixed effects, CuPy works with sparse matrices and the sparse LSMR solver (as is e.g. [available in scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)).

This strategy is amenable for GPU acceleration, and for problems where the standard demeaner struggles to converge, this strategy can lead to significant speedups if paired with a GPU.

![Complex Fixed Effects Structure: Benchmark](https://raw.githubusercontent.com/py-econometrics/pyfixest/master/benchmarks/complex_benchmarks.png)

Note that for smaller and more well-behaved problems, running the alternating projections algorithm on the CPU via `numba` or `rust` usually seems to work better: 

![Simply Fixed Effects Structure: Benchmark](https://raw.githubusercontent.com/py-econometrics/pyfixest/master/benchmarks/simple_benchmarks_cupy.png)

**Benchmark Hardware Specifications:**
- **CPU**: x86_64, 8 physical cores @ 3.2 GHz, 44 GB RAM
- **GPU**: NVIDIA RTX A6000, 48 GB memory, Compute Capability 8.6

## Installation

To run pyfixest via cup on the GPU, you need to install the required dependency: 

```bash
# For CUDA 11.x, 12.x, 13.x
pip install cupy-cuda11x
pip install cupy-cuda12x
pip install cupy-cuda13x
```