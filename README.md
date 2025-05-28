
![kifit_logo](https://github.com/user-attachments/assets/950adab6-bc29-4850-b916-57bb902bd8cd)


Searching for New Physics with (Non-)Linear King Plots



# Installation instructions

To install the package run please clone the repository, enter it and type:
```
pip install .
```

# Example

An example of usage of `kifit` can be found in the [`main.py`](https://github.com/MatteoRobbiati/kifit/blob/main/src/kifit/main.py) file.

It runs a full `kifit` pipeline:
1. ⚛️ data loading;
2. 🕵️ determine an optimal $\alpha_{\rm NP}$ search window;
3. 🧑‍🔬 experiments to determine the bounds.

A `kifit` execution can be customized through various parameters.
A standard setup (with light Monte Carlo sampling) to run the `main.py` file can be found in the following code block:

```sh
python3 main.py --element_list "Ca_WT_Aarhus_PTB_2024"\
                --num_alphasamples_search 100\
                --num_elemsamples_per_alphasample_search 100 \
                --search_mode "detlogrid"\
                --logrid_frac 2\
                --num_exp 20\
                --block_size 5\
                --min_percentile 1\
                --num_sigmas 2\
                --num_alphasamples_exp 100 \
                --num_elemsamples_exp 100 \
                --x0_fit 0 \
                --gkp_dims 3\
                --proj_dims 3\
                --nmgkp_dims 3\
                --num_det_samples 1000 \
                --x0_det 0\
                --showalldetbounds \
                --showbestdetbounds \
                --verbose
```


# The code in a diagram

In the following diagram you can find a schematic overview of the `kifit` algorithm.
![kifit_diagram](https://github.com/user-attachments/assets/0d9c918f-eef3-4fc1-ab5c-5845907d40b1)

