#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "worst_Yb_Kyoto_MIT_GSI_PTB_2024"\
                --optimization_method "Powell" \
                --maxiter 600 \
                --num_searches 20 \
                --num_elemsamples_search 500 \
                --num_exp 20\
                --block_size  10\
                --min_percentile 5\
                --num_sigmas 2\
                --num_alphasamples_exp 500 \
                --num_elemsamples_exp 500 \
                --x0_fit $(seq 0 25 800) \
                --gkp_dims 3\
                --num_det_samples 500 \
                --x0_det $(seq 0 25 800)\
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
