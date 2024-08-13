#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit_%A_%a.log  
#SBATCH --array=0-98

# Define the element list
ELEMENT_LIST=("best_Yb_Kyoto_MIT_GSI_PTB_2024" "worst_Yb_Kyoto_MIT_GSI_2022" "best_Yb_Kyoto_MIT_GSI_2022")

# Define the x0_fit values array in bash
X0_FIT_VALUES=($(seq 0 25 800))

# Calculate the total number of tasks needed
NUM_X0_VALUES=${#X0_FIT_VALUES[@]}
NUM_ELEMENTS=${#ELEMENT_LIST[@]}


# Calculate the index for element_list and x0_fit
ELEMENT_INDEX=$(($SLURM_ARRAY_TASK_ID / NUM_X0_VALUES))
X0_FIT_INDEX=$(($SLURM_ARRAY_TASK_ID % NUM_X0_VALUES))

# Get the corresponding element and x0_fit value
ELEMENT=${ELEMENT_LIST[$ELEMENT_INDEX]}
X0=${X0_FIT_VALUES[$X0_FIT_INDEX]}

python3 main.py --element_list "$ELEMENT" \
                --optimization_method "Powell" \
                --maxiter 600 \
                --num_searches 20 \
                --num_elemsamples_search 500 \
                --num_exp 20 \
                --block_size 10 \
                --min_percentile 5 \
                --num_sigmas 2 \
                --num_alphasamples_exp 500 \
                --num_elemsamples_exp 500 \
                --gkp_dims 3 \
                --num_det_samples 500 \
                --x0_fit $X0 \
                --x0_det $X0 \
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
