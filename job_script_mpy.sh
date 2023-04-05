#!/bin/bash

#SBATCH -A training2302
#SBATCH --partition=dp-cn
#SBATCH --job-name=mpi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tyl1@hi.is

#load modules

module load Stages/2020  
module load GCC/10.3.0  
module load ParaStationMPI/5.4.9-1
module load GCCcore/.11.2.0
module load Python/3.8.5
module load matplotlib/3.4.3
module load tqdm/4.62.3
module load mpi4py/3.0.3-Python-3.8.5



#run Python program
srun --cpu-bind=none python -u mpi.py
