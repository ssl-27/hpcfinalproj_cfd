#!/bin/bash

#SBATCH -A training2302
#SBATCH --partition=dp-cn
#SBATCH --job-name=1500_481
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tyl1@hi.is

#load modules

module load Stages/2022
module load GCCcore/.11.2.0
module load Python/3.9.6
module load matplotlib/3.4.3
module load tqdm/4.62.3



#run Python program
srun --cpu-bind=none python -u Lid_Driven_Cavity1500_481.py
