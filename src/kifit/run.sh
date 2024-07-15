#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --elements_list "DataCa,DataYb" \
                --outputfile_name "PippoMPHIVAR" --optimization_method "Powell" \
                --maxiter 100 --num_searches 5 --num_elemsamples_search 200 \
                --num_experiments 5  --block_size  10\
                --num_alphasamples_exp 100 --num_elemsamples_exp 100 \
                --num_samples_det 100 --mphivar "true"
                     
                     