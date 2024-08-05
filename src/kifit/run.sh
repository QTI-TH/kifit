#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log

python3 main.py --element_list "Camin"\
                --optimization_method "Powell" \
                --maxiter 100 \
                --num_searches 5 \
                --num_elemsamples_search 100 \
                --num_exp 5\
                --block_size  10\
                --min_percentile 5\
                --num_sigmas 2\
                --num_alphasamples_exp 100 \
                --num_elemsamples_exp 100 \
                --gkp_dims 3\
                --num_det_samples 100\
                --x0_fit 0 \
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
