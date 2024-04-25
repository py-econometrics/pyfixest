## Readme

All benchmarks follow `fixest`'s benchmarks, which you can find [here](https://github.com/lrberge/fixest/tree/master/_BENCHMARK).
`PyFixest` benchmarks were run on a Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz, 2304Mhz, 4 Core(s), 8 Logical Processor(s).
Timings for `R`, `Stata` and `Julia` programs are taken from the `fixest`'s benchmarks.

To run the python benchmarks, you need to install the following packages:
    - `pyfixest`
    - `pandas`
    - `numpy`
    - `tqdm`
First, you need to create the data by running the `data_generation.R` files. This will populate the `_STATA` and `data` folders.
Then, you can run the `run_benchmarks.ipynb` notebook to run the benchmarks. This will populate the `results_py.csv` file.
Finally, you can run the `plot_benchmarks.ipynb` notebook to generate the plots.
