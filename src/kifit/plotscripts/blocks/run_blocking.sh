#!/bin/bash
#SBATCH --job-name=blocks
#SBATCH --output=blocking.log

N_ALPHA=200
N_ELEMS=200
N_EXPS=2000
BLOCK_SIZE=40
X0=0

python3 main.py --element_list "Ca_WT_Aarhus_PTB_2024"\
                --num_alphasamples_search $N_ALPHA\
                --num_elemsamples_per_alphasample_search $N_ELEMS \
                --search_mode "detlogrid"\
                --logrid_frac 5\
                --num_exp $N_EXPS\
                --block_size $BLOCK_SIZE\
                --min_percentile 2\
                --num_sigmas 2\
                --num_alphasamples_exp $N_ALPHA \
                --num_elemsamples_exp $N_ELEMS \
                --x0_fit $X0 \
                --gkp_dims 3\
                --proj_dims 3\
                --nmgkp_dims 3\
                --num_det_samples 1000 \
                --x0_det $X0\
                --showalldetbounds \
                --showbestdetbounds \
                --verbose

