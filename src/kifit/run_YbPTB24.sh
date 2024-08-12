#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Yb_PTB_2024"\
                --optimization_method "Powell" \
                --maxiter 100 \
                --num_searches 10 \
                --num_elemsamples_search 200 \
                --num_exp 10\
                --block_size  10\
                --min_percentile 5\
                --num_sigmas 2\
                --num_alphasamples_exp 200 \
                --num_elemsamples_exp 200 \
                --x0_fit $(seq 0 25 800) \
                --gkp_dims 3\
                --num_det_samples 200 \
                --x0_det $(seq 0 25 800)\
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
