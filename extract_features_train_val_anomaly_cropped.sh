#!/bin/bash
#SBATCH --account=def-panos
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --mail-user=aisha.eldeeb.ubc@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=train_val_anomaly_cropped_feature_extraction_output.out

# Load required modules
module load StdEnv/2020 cuda/11.4 cudnn/8.2.0 llvm/8 python/3.8 geos/3.8.1
export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$CUDA_HOME/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.4/cudnn/8.2.0/lib64
export LLVM_CONFIG=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/llvm/8.0.1/bin/llvm-config
export NCCL_BLOCKING_WAIT=1

# Define paths
BASE_INPUT="/home/aishaeld/scratch/dataset/videos"
BASE_OUTPUT="/home/aishaeld/scratch/dataset/features"
ENV_NAME="feature_extraction_env"
DIR_NAME="I3D_Feature_Extraction_resnet"

# Navigate to project directory
cd /home/$USER/scratch/$DIR_NAME

# Activate virtual environment
source $ENV_NAME/bin/activate

# Function to extract features
extract_features() {
    local input_path=$1
    local output_path=$2

    # Check if input directory exists
    if [ ! -d "$input_path" ]; then
        echo "Error: Input directory does not exist: $input_path"
        return 1
    fi

    # Ensure output directory exists
    mkdir -p "$output_path"

    # Log the operation
    echo "Extract features from $input_path to $output_path"

    # Run feature extraction
    python main.py --datasetpath="$input_path" --outputpath="$output_path"

    # Check if the extraction was successful
    if [ $? -ne 0 ]; then
        echo "Error: Feature extraction failed for $input_path"
        return 1
    fi
}

# Perform feature extraction for all datasets

extract_features "$BASE_INPUT/train_val/anomaly_cropped" "$BASE_OUTPUT/train_val/anomaly_cropped"

echo "Feature extraction completed!"


