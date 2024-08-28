#!/bin/bash
#SBATCH --job-name=Ca2015_ng
#SBATCH --output=kifit_%A_%a.log  
#SBATCH --array=0-32 

# Define the x0_fit values array in bash
X0_FIT_VALUES=($(seq 0 25 800))

# Get the x0_fit value corresponding to this array task
X0=${X0_FIT_VALUES[$SLURM_ARRAY_TASK_ID]}

python3 main.py --element_list "Ca_PTB_2015" \
                --optimization_method "Powell" \
                --maxiter 1000 \
                --num_searches 20 \
                --num_elemsamples_search 400 \
                --num_exp 50 \
                --block_size 5 \
                --min_percentile 5 \
                --num_sigmas 2 \
                --num_alphasamples_exp 400 \
                --num_elemsamples_exp 400 \
                --x0_fit $X0 \
                --x0_det $X0 \
                --showalldetbounds \
                --verbose
