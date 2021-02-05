#!/bin/bash -l
#SBATCH --job-name=threshold_3
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=threshold_3.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_04022021.sif python3 threshold_3.py

command time -v srun --export=all -N 1 -n 1 singularity exec quop_mpi_04022021.sif python3 process_threshold_3.py

