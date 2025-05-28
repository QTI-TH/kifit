#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Ca_WT_Aarhus_PTB_2024"\
                --num_alphasamples_search 100\
                --num_elemsamples_per_alphasample_search 100 \
                --search_mode "detlogrid"\
                --logrid_frac 2\
                --num_exp 20\
                --block_size 5\
                --min_percentile 1\
                --num_sigmas 2\
                --num_alphasamples_exp 100 \
                --num_elemsamples_exp 100 \
                --x0_fit 0 \
                --gkp_dims 3\
                --proj_dims 3\
                --nmgkp_dims 3\
                --num_det_samples 1000 \
                --x0_det 0\
                --showalldetbounds \
                --showbestdetbounds \
                --verbose