#!/bin/bash -l
#SBATCH --job-name=3node_qwoa
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --account=pawsey0309
#SBATCH --output=3node_qwoa.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_06012021.sif python3 benchmark.py

command time -v srun --export=all -N 1 -n 1 singularity exec quop_mpi_06012021.sif python3 process_data.py

