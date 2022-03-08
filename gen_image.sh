#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=46
#SBATCH -N 1
#SBATCH -n 100
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH -o job4.o
#SBATCH -e job4.e
#SBATCH -t 6-0
/bin/bash
conda activate hetero_mod
module load python/3.8.9
module load openmpi/4.1.0/gcc.7.3.1/rocm.4.2


srun --exclusive --nodes 1 --ntasks 1 python dataset_test.py
wait
