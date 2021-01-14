#!/bin/bash -l
#SBATCH --job-name=8node_qwoa
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=8node_qwoa.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_06012021.sif python3 p200.py

command time -v srun --export=all -N 1 -n 1 singularity exec quop_mpi_06012021.sif python3 process_p200.py

