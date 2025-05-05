#!/bin/bash
#SBATCH --job-name=det_Yb
#SBATCH --output=kifit.log


python3 main.py --element_list "Yb_Kyoto_MIT_GSI_PTB_2024"\
                --num_sigmas 2\
                --gkp_dims 3\
                --nmgkp_dims 3\
                --proj_dims 3\
                --num_det_samples 5000\
                --x0_det $(seq 0 25 800)\
                --showbestdetbounds\
                --showalldetbounds\
                --verbose
