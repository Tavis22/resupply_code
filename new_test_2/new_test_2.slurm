#!/bin/bash -l
#SBATCH --job-name=new_test_2
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0309
#SBATCH --output=new_test_2.out

export OMP_NUM_THREADS=1

module load singularity

command time -v srun --export=all -N 1 -n 24 singularity exec quop_mpi_18012021.sif python3 new_test_2.py

command time -v srun --export=all -N 1 -n 1 singularity exec quop_mpi_18012021.sif python3 new_test_2_process_data.py

