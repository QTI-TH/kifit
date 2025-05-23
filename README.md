
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
2. 🕵️ search of the best $\alpha_{\rm NP}$ window;
3. 🧑‍🔬 experiments to determine the bounds.

A possible way to execute `main.py` can be found in the following code block:

```sh
python3 main.py --elements_list "Camin,Ybmin" \
                --outputfile_name "CaminYbmin" --optimization_method "Powell" \
                --maxiter 1000 --num_searches 20 --num_elemsamples_search 300 \
                --num_experiments 20  --block_size  10\
                --num_alphasamples_exp 200 --num_elemsamples_exp 300 \
                --num_samples_det 100 --mphivar "true"
```


# The code in a diagram

In the following diagram you can find a schematic overview of the `kifit` algorithm.
![kifit_diagram](https://github.com/user-attachments/assets/0d9c918f-eef3-4fc1-ab5c-5845907d40b1)

