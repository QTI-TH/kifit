#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --elements_list "Ybmin" \
                --outputfile_name "PippoTest" --optimization_method "Powell" \
                --maxiter 1000 --num_searches 2 --num_elemsamples_search 100 \
                --num_experiments 1  --block_size 1 \
                --num_alphasamples_exp 100 --num_elemsamples_exp 100 \
                --num_samples_det 100
                     
                     