# Customized I3D Feature Extraction with ResNet

## Overview
This repository is a customized version of [I3D Feature Extraction with ResNet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet). The changes were made to suit the needs of a structured dataset processed on **Compute Canada**. It efficiently extracts video features using ResNet and provides modular scripts for debugging and flexibility.

---

## Changes Made

### Changes to `extract_features.py`
The `extract_features.py` script was customized to improve functionality and compatibility. Key changes include:

1. **Skipping Insufficient Frames**  
   - Added a check to skip directories with insufficient frames:
     ```python
     if frame_cnt <= chunk_size:
         print(f"Skipping {frames_dir}: Insufficient frames ({frame_cnt})")
         return None
     ```

2. **Enhanced Frame Clipping**  
   - Adjusted frame clipping logic for compatibility with smaller datasets.

3. **Efficient Batch Processing**  
   - Used `torch.no_grad()` for efficient GPU memory usage during inference.

4. **Refined Feature Output**  
   - Ensured extracted features meet downstream requirements:
     ```python
     full_features = full_features[:,:,:,0,0,0]
     full_features = np.array(full_features).transpose([1, 0, 2])
     ```

5. **Improved Logging**  
   - Added logs for runtime visibility:
     ```python
     print("batchsize", batch_size)
     ```

---

### Changes to `main.py`
The `main.py` script was enhanced to improve usability and user experience. Key updates include:

1. **Summary Logging**  
   - Logs run details in `feature_config.txt`:
     ```json
     {
         "run_date": "2024-12-02 12:00:00",
         "total_videos": 100,
         "videos_skipped": 5,
         "processed_videos": 95
     }
     ```

2. **Graceful Skipping of Videos**  
   - Skips videos with insufficient frames and logs the issue.

3. **Feature Metadata Collection**  
   - Logs metadata for extracted features (video name, file path, dimensions).

4. **Temporary Directory Management**  
   - Cleans up temporary directories after processing.

5. **Flexible Output Paths**  
   - Automatically creates nested directories for outputs.

6. **Enhanced Logs**  
   - Added detailed logs for each video, including processing time.

---

## Platform and Environment

- **Platform**: Compute Canada (Narval cluster)
- **Environment**:  
  - Use `job_scripts/install_packages.sh` to install dependencies (CUDA, cuDNN, Python packages).
  - Create and activate a virtual environment before running:
    ```bash
    source job_scripts/install_packages.sh
    ```

---

## Dataset and Features Structure

The dataset is structured as follows, ensuring traceability between input videos and output features:

```bash
dataset/
├── videos/
│   ├── train_val/
│   │   ├── anomaly/
│   │   ├── anomaly_augmented/
│   │   ├── anomaly_cropped/
│   │   ├── non_anomaly/
│   │   ├── non_anomaly_augmented/
│   │   ├── non_anomaly_cropped/
│   └── test/
│       ├── anomaly/
│       ├── anomaly_augmented/
│       ├── anomaly_cropped/
│       ├── non_anomaly/
│       ├── non_anomaly_cropped/
└── features/ # Mirrors the videos/ structure
```

## Included Scripts

### Job Scripts
The `job_scripts/` folder contains SLURM job scripts for modular feature extraction from each dataset subfolder. This approach allows for better debugging and granular processing.

#### Example Scripts
- `extract_features_train_val_anomaly.sh`
- `extract_features_train_val_anomaly_cropped.sh`
- `extract_features_test_non_anomaly.sh`

#### SLURM Script Example
Below is an example SLURM script for extracting features from the `train_val/anomaly_augmented` folder:

```bash
#!/bin/bash
#SBATCH --account=def-panos
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --output=train_val_anomaly_augmented_feature_extraction_output.out

# Load necessary modules
module load StdEnv/2020 cuda/11.4 cudnn/8.2.0 llvm/8 python/3.8

# Activate the virtual environment
source feature_extraction_env/bin/activate

# Run feature extraction
python main.py --datasetpath="/home/$USER/scratch/dataset/videos/train_val/anomaly_augmented" \
               --outputpath="/home/$USER/scratch/dataset/features/train_val/anomaly_augmented"
```
## How to Run

### Setup Environment
Install dependencies and set up the virtual environment:
```bash
source job_scripts/install_packages.sh
```
### Run Feature Extraction
Submit batch jobs for specific subfolders:
```bash
sbatch job_scripts/extract_features_train_val_anomaly.sh
```

###Check Logs
Logs and output files for each job are saved in .out files in the current directory.
               

The extracted features from this repository are used for model training. See the Model [Training Repository]() for more details. Ensure the features directory mirrors the dataset structure for seamless integration.