#!/bin/bash -l
#SBATCH --job-name=multirun_p20
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=multirun_p20.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_18012021.sif python3 multirun_p20.py

command time -v srun --export=all -N 1 -n 1 singularity exec quop_mpi_18012021.sif python3 process_multirun_p20.py

