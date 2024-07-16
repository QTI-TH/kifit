#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit_%A_%a.log
#SBATCH --array=1-3


num_searches_list=(5 10 20)
num_searches=${num_searches_list[$SLURM_ARRAY_TASK_ID-1]}

# num_elemsamples_list=(50 100 200 500 1000)
# num_elemsamples=${num_elemsamples_list[$SLURM_ARRAY_TASK_ID-1]}

python3 main.py --outputfile_name "testing_bounds_${SLURM_ARRAY_TASK_ID}" \
                --elements_list "Yb_Kyoto_MIT_GSI_PTB_2024,Ca_WT_Aarhus_PTB_2024" \
                --optimization_method "Powell" \
                --maxiter 1000 --num_searches $num_searches --num_elemsamples_search 200 \
                --num_experiments 5  --block_size  5 \
                --num_alphasamples_exp 200 --num_elemsamples_exp 200 \
                --num_samples_det 100 --random_seed 42 --x0 0 \
                     
                     