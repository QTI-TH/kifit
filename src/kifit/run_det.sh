#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Camin"\
                --num_sigmas 2\
                --gkp_dims 3\
                --num_det_samples 1000\
                --x0_det $(seq 0 25 800)\
                --showbestdetbounds\
                --showalldetbounds\
                --verbose
