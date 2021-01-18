#!/bin/bash -l
#SBATCH --job-name=basin_tests
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=basin_tests.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_18012021.sif python3 scaled_basin_tests.py

comma

