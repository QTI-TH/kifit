#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --elements_list "Ybmin,Camin" \
                --outputfile_name "Combined2" --optimization_method "Powell" \
                --maxiter 1000 --num_searches 10 --num_elemsamples_search 100 \
                --num_experiments 10  --block_size  10\
                --num_alphasamples_exp 200 --num_elemsamples_exp 200 \
                --num_samples_det 100
                     
                     