#!/bin/bash -l

#SBATCH -p shared-gpu
#SBATCH -N 1
#SBATCH -C gpu_count:8
#SBATCH --qos=long
#SBATCH --time=2-00:00:00

conda activate anujnlp
python semi_supervised.py 4 5 semisup20_2models.txt validation20_2models.txt 0.45
