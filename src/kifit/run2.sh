#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit_%A_%a.log  
#SBATCH --array=0-23  # Updated to reflect the total number of tasks


# Define the x0 values array in bash
X0_VALUES=(0 400 800)

# Define the num_searches array
NUM_SEARCHES_VALUES=(5 10 20 50 100 200 500 1000)

# Calculate the number of values in each array
NUM_X0_VALUES=${#X0_VALUES[@]}
NUM_SEARCHES=${#NUM_SEARCHES_VALUES[@]}

# Calculate the total number of tasks needed
TOTAL_TASKS=$(($NUM_X0_VALUES * $NUM_SEARCHES))

# Calculate the index for x0 and num_searches
X0_INDEX=$(($SLURM_ARRAY_TASK_ID / $NUM_SEARCHES))
NUM_SEARCHES_INDEX=$(($SLURM_ARRAY_TASK_ID % $NUM_SEARCHES))

# Get the corresponding x0 value and num_searches value
X0=${X0_VALUES[$X0_INDEX]}
NUM_SEARCHES=${NUM_SEARCHES_VALUES[$NUM_SEARCHES_INDEX]}

# Run the Python script with the specified parameters
python3 main.py --element_list "Ca_WT_Aarhus_PTB_2024" \
                --optimization_method "Powell" \
                --maxiter 600 \
                --num_searches $NUM_SEARCHES \
                --num_elemsamples_search 200 \
                --num_exp 20 \
                --block_size 10 \
                --min_percentile 5 \
                --num_sigmas 2 \
                --num_alphasamples_exp 200 \
                --num_elemsamples_exp 200 \
                --gkp_dims 3 \
                --num_det_samples 1000 \
                --x0_fit $X0 \
                --x0_det $X0 \
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
