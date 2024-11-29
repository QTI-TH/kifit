#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Ca_WT_Aarhus_PTB_2024"\
                --optimization_method "Powell" \
                --maxiter 1000 \
                --init_globalopt\
                --num_searches 20 \
                --num_elemsamples_search 400 \
                --num_exp 50\
                --block_size  5\
                --min_percentile 5\
                --num_sigmas 2\
                --num_alphasamples_exp 400 \
                --num_elemsamples_exp 400 \
                --x0_fit $(seq 0 25 800) \
                --gkp_dims 3\
                --nmgkp_dims 3\
                --proj_dims 3\
                --num_det_samples 1000 \
                --x0_det $(seq 0 25 800)\
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
