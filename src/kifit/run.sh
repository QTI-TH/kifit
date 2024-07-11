#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --elements_list "Yb_PTB_2024,Ca_WT_Aarhus_PTB_2024" \
                --outputfile_name "PippoCaminetto" --optimization_method "Powell" \
                --maxiter 1000 --num_searches 10 --num_elemsamples_search 100 \
                --num_experiments 5  --block_size  10\
                --num_alphasamples_exp 100 --num_elemsamples_exp 100 \
                --num_samples_det 100
                     
                     