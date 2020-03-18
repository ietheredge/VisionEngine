#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Node feature:
#SBATCH --constraint="gpu"
# Specify number of GPUs to use:
#SBATCH --gres=gpu:rtx5000:1            # If using only 1 GPU of a shared node
#SBATCH --mem=92500           # Memory is necessary if using only 1 GPU
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20    # If using only 1 GPU of a shared node
#
#SBATCH --mail-type=none
#SBATCH --mail-user=ietheredge@ab.mpg.de
#
# wall clock limit:
#SBATCH --time=24:00:00

module load cuda/10.0 cudnn/7.4 nccl/2.3.7 anaconda/3/5.1

# Run the program:

python VisionEngine/main.py -c VisionEngine/configs/butterfly_vae_config_mmd5000_nokl_wpercep_reconp5.json > log2.log