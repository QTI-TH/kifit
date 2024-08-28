#!/bin/bash
#SBATCH --job-name=kif_elem
#SBATCH --output=kifit_%A_%a.log  
#SBATCH --array=1-6

# Define the different values for num_elemsamples_search
elemsamples_values=(50 100 200 500 1000 5000)

# Get the value corresponding to the current array task ID
num_elemsamples_search=${elemsamples_values[$SLURM_ARRAY_TASK_ID-1]}

python3 main.py --element_list "Ca_WT_Aarhus_PTB_2024" \
                --optimization_method "Powell" \
                --maxiter 1000 \
                --num_searches 20 \
                --num_elemsamples_search $num_elemsamples_search \
                --num_exp 20 \
                --block_size 10 \
                --min_percentile 5 \
                --num_sigmas 2 \
                --num_alphasamples_exp 200 \
                --num_elemsamples_exp $num_elemsamples_search \
                --gkp_dims 3 \
                --num_det_samples 1000 \
                --x0_fit 400 \
                --x0_det 400 \
                --showbestdetbounds \
                --showalldetbounds \
                --verbose

