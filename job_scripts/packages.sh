#!/bin/bash
#SBATCH --account=def-panos
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --mail-user=aisha.eldeeb.ubc@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=%j-%x.log
 
module load StdEnv/2020 cuda/11.4 cudnn/8.2.0 llvm/8 python/3.8 geos/3.8.1
export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$CUDA_HOME/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.4/cudnn/8.2.0/lib64
export LLVM_CONFIG=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/llvm/8.0.1/bin/llvm-config
# module load StdEnv/2020 cuda/11.1 cudnn/8.2.0 llvm/8 python/3.7 geos/3.8.1
# export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$CUDA_HOME/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.1/cudnn/8.2.0/lib64
 
# module load nixpkgs/16.09 gcc/7.3.0 cuda/10.1 cudnn/7.6.5 python/3.7 geos/3.7.2
# export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$CUDA_HOME/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2017/CUDA/cuda10.1/cudnn/7.6.5/lib64
 
export NCCL_BLOCKING_WAIT=1
export ENV_NAME=env3
export TMPDIR=$SLURM_TMPDIR
 
cd /home/$USER &&
rm -Rf /home/$USER/$ENV_NAME &&
rm -Rf /home/$USER/.cache/pip &&
source /home/$USER/$ENV_NAME/bin/activate &&
 
pip install --no-index visdom
