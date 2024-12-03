# Customized I3D Feature Extraction with ResNet

## Overview
This repository is a customized version of [I3D Feature Extraction with ResNet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet). The changes were made to suit the needs of the **[Smart Surveillance](https://github.com/aishaeldeeb/smart_surveillance)** project. It efficiently extracts video features using ResNet and provides modular scripts for debugging and flexibility.

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

#### Platform
These scripts were designed to run on **Compute Canada - Narval cluster**

#### SLURM Job Script Example
```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT_NAME>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --output=train_val_anomaly_augmented.out
```

## How to Run

### Setup
- Set up the virtual environment
- Install dependencies using `requirements.txt`

### Run Feature Extraction
#### Using a Job Script
Submit batch jobs for specific subfolders within the dataset root directory:
```bash
sbatch job_scripts/extract_features_train_val_anomaly.sh
```
#### Without a Job Script
Run the feature extraction directly:
```python
python main.py --datasetpath="/home/$USER/scratch/dataset/videos/train_val/anomaly_augmented" \
               --outputpath="/home/$USER/scratch/dataset/features/train_val/anomaly_augmented"

```
### Check Logs
Logs and output files for each job are saved in .out files in the current directory.
               

