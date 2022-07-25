#!/bin/bash -l

#SBATCH -p shared-gpu
#SBATCH -N 1
#SBATCH -C gpu_count:8
#SBATCH --qos=long
#SBATCH --time=2-00:00:00

conda activate anujnlp
python semi_supervised.py 4 5 semi_sup_otpt_20.txt validation_otpt_20.txt 0.45
