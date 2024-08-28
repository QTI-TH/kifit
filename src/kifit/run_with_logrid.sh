#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Camin"\
                --num_alphasamples_search 1000\
                --num_elemsamples_per_alphasample_search 100 \
                --logrid_frac -2\
                --num_exp 20\
                --block_size  10\
                --min_percentile 5\
                --num_sigmas 2\
                --num_alphasamples_exp 200 \
                --num_elemsamples_exp 200 \
                --x0_fit $(seq 0 400 800) \
                --gkp_dims 3\
                --num_det_samples 1000 \
                --x0_det $(seq 0 400 800)\
                --showbestdetbounds \
                --showalldetbounds \
                --showalldetvals \
                --verbose
