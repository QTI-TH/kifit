#!/bin/bash
#SBATCH --job-name=blocking
#SBATCH --output=kifit_%A_%a.log  

python3 main.py --element_list "Yb_PTB_2024"\
                --optimization_method "Powell" \
                --maxiter 1000 \
                --num_searches 20 \
                --num_elemsamples_search 200 \
                --num_exp 10 \
                --block_size 200 \
                --min_percentile 5 \
                --num_sigmas 2 \
                --num_alphasamples_exp 200 \
                --num_elemsamples_exp 200 \
                --gkp_dims 3 \
                --num_det_samples 1000 \
                --x0_fit 0 \
                --x0_det 0 \
                --showbestdetbounds \
                --showalldetbounds \
                --init_globalopt \
                --verbose
