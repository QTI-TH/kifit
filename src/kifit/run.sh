#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Camin" \
                --optimization_method "Powell" \
                --maxiter 100 \
                --num_searches 5 \
                --num_elemsamples_search 200 \
                --num_exp 5\
                --block_size  10\
                --min_percentile 1\
                --num_sigmas 2\
                --num_alphasamples_exp 100\
                --num_elemsamples_exp 100 \
                --num_det_samples 100\
                --gkp_dims 3\
                --x0 0\
                --verbose
