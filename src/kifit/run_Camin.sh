#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Camin"\
                --num_alphasamples_search 1000\
                --num_elemsamples_per_alphasample_search 100 \
                --logrid_frac -5\
                --num_exp 20\
                --block_size  50\
                --min_percentile 0\
                --num_sigmas 1\
                --num_alphasamples_exp 1000 \
                --num_elemsamples_exp 1000 \
                --x0_fit 0 \
                --gkp_dims 3\
                --proj_dims 3\
                --num_det_samples 1000 \
                --x0_det 0\
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
