#!/bin/bash
#SBATCH --job-name=kifit
#SBATCH --output=kifit.log


python3 main.py --element_list "Ca_WT_Aarhus_2024"\
                --num_sigmas 2\
                --gkp_dims 3\
                --num_det_samples 1000\
                --x0_det 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\
                --showbestdetbounds\
                --showalldetbounds\
                --verbose
