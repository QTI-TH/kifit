#!/bin/bash
#SBATCH --job-name=kifit_es
#SBATCH --output=kifit_%A_%a.log
#SBATCH --array=1-5


# num_searches_list=(5 10 20 50 100 500 1000)
# num_searches=${num_searches_list[$SLURM_ARRAY_TASK_ID-1]}

num_elemsamples_list=(50 100 200 500 1000)
num_elemsamples=${num_elemsamples_list[$SLURM_ARRAY_TASK_ID-1]}

python3 main.py --outputfile_name "nelems_${SLURM_ARRAY_TASK_ID}" \
                --elements_list "Yb_Kyoto_MIT_GSI_PTB_2024,Ca_WT_Aarhus_PTB_2024" \
                --optimization_method "Powell" \
                --maxiter 1000 --num_searches 20 --num_elemsamples_search $num_elemsamples \
                --num_experiments 10  --block_size  5 \
                --num_alphasamples_exp 200 --num_elemsamples_exp $num_elemsamples \
                --num_samples_det 100 --random_seed 42 --x0 0 \
                     
                     