#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Ca_WT_Aarhus_PTB_2024"\
                --num_alphasamples_search 500\
                --num_elemsamples_per_alphasample_search 500 \
                --logrid_frac -5\
                --num_exp 50\
                --block_size  10\
                --min_percentile 5\
                --num_sigmas 2\
                --num_alphasamples_exp 500 \
                --num_elemsamples_exp 500 \
                --x0_fit $(seq 0 25 800) \
                --gkp_dims 3\
                --nmgkp_dims 3\
                --proj_dims 3\
                --num_det_samples 1000 \
                --x0_det $(seq 0 25 800)\
                --showalldetbounds \
                --showbestdetbounds \
                --verbose
