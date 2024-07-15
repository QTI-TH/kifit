#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --outputfile_name "test_run" \
                --elements_list "DataCa" \
                --optimization_method "Powell" \
                --maxiter 100 --num_searches 2 --num_elemsamples_search 100 \
                --num_experiments 2  --block_size  5\
                --num_alphasamples_exp 100 --num_elemsamples_exp 100 \
                --num_samples_det 100 --mphivar "true"
                     
                     