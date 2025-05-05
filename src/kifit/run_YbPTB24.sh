#!/bin/bash
#SBATCH --job-name=Yb24
#SBATCH --output=kifit_%A_%a.log
#SBATCH --array=0-32  

N_ALPHA=50
N_ELEMS=50
N_EXPS=4
BLOCK_SIZE=2

# Parameters
X0_VALUES=($(seq 0 25 800))  
X0=${X0_VALUES[$SLURM_ARRAY_TASK_ID]}  

python3 main.py --element_list "Yb_Kyoto_MIT_GSI_PTB_MPIK_2024"\
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
                --num_det_samples 5000 \
                --x0_det $X0\
                --showalldetbounds \
                --showbestdetbounds \
                --verbose