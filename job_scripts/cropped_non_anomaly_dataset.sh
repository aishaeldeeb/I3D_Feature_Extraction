#!/bin/bash
#SBATCH --account=def-panos
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --mail-user=aisha.eldeeb.ubc@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=cropped_output2.out

root=/home/aishaeld/

module load StdEnv/2020
module load gentoo/2020
module load python/3.7.7
module load intel/2020.1.217  cuda/11.4


source $root/env3/bin/activate
echo "Extract Features"
python ../main.py --datasetpath=train/non_anomaly_cropped --outputpath=features_v3/non_anomaly
echo "End"
