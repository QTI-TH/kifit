#!/bin/bash
#SBATCH --job-name=det_meth
#SBATCH --output=kifit_%A_%a.log
#SBATCH --array=0-32  # Create an array job with indices 0 to 32

# Calculate the x0_det value for this array task
x0=$((SLURM_ARRAY_TASK_ID * 25))

python3 main.py --element_list "Ca_PTB_2015" \
                --num_sigmas 2 \
                --gkp_dims 3 \
                --nmgkp_dims 3 \
                --num_det_samples 1000 \
                --x0_det $x0 \
                --showbestdetbounds \
                --showalldetbounds \
                --verbose
