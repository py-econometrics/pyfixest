## `PyFixest` on professional-tier GPUs 

`PyFixest` allows to run the fixed effects demeaning on the GPU via the `demeaner_backend` argument. 
To do so, you will have to install `jax` and `jaxblib`, for example by typing `pip install pyfixest[jax]`.

We test two back-ends for the iterative alternating-projections component of the fixed-effects regression on an Nvidia A100 GPU with 40 GB VRAM (a GPU that one typically wouldn't have installed to play graphics-intensive videogames on consumer hardware). `numba` benchmarks are run on a 12-core xeon CPU. 

The JAX backend exhibits major performance improvements **on the GPU** over numba in large problems. 

![](figures/gpu_benchmarks.png)


On the **CPU** instead, we find that `numba` outperforms the JAX backend. You can find details in the [benchmark section](https://github.com/py-econometrics/pyfixest/tree/master/benchmarks) of the github repo. 