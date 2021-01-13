#!/bin/bash -l
#SBATCH --job-name=quop_mpi_example
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --account=pawsey0309
#SBATCH --output=example.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_06012021.sif python3 construct6nodeQual.py
