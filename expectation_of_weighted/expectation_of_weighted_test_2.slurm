#!/bin/bash -l
#SBATCH --job-name=expectation_of_weighted_2
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=expectation_of_weighted_test_2.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_04022021.sif python3 expectation_of_weighted_test_2.py

command time -v srun --export=all -N 1 -n 1 singularity exec quop_mpi_04022021.sif python3 process_expectation_of_weighted_test_2.py

