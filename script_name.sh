#!/bin/bash -l

#SBATCH -p shared-gpu
#SBATCH -N 1
#SBATCH -C gpu_count:8
#SBATCH --qos=long
#SBATCH --time=2-00:00:00

conda activate anujnlp
python semi_supervised.py 3 20 semi_sup_otpt_60.txt validation_otpt_60.txt 0.45
