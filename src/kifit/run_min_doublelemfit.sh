#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Camin" "Camin_swap"\
                --num_alphasamples_search 100\
                --num_elemsamples_per_alphasample_search 100 \
                --search_mode "detlogrid"\
                --logrid_frac 5\
                --num_exp 2\
                --block_size  50\
                --min_percentile 1\
                --num_sigmas 2\
                --num_alphasamples_exp 100 \
                --num_elemsamples_exp 100 \
                --x0_fit 500 \
                --gkp_dims 3 \
                --x0_det 500 \
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
