# Customized I3D Feature Extraction with ResNet

## Overview
This repository is a customized version of [Original Repo](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet). The changes were made to better suit the needs of [Your Dataset or Task]. It extracts video features efficiently using ResNet.

## Changes Made

### Changes to `extract_features.py`

The `extract_features.py` script was customized to improve functionality and flexibility. Key changes include:

1. **Skipping Insufficient Frames**  
   - Added a check to skip directories with insufficient frames, avoiding abrupt termination:
     ```python
     if frame_cnt <= chunk_size:
         print(f"Skipping {frames_dir}: Insufficient frames ({frame_cnt})")
         return None
     ```

2. **Enhanced Frame Clipping**  
   - Adjusted frame clipping logic for better compatibility with smaller datasets.

3. **Efficient Batch Processing**  
   - Retained `torch.no_grad()` for efficient GPU memory usage during inference.

4. **Refined Feature Output**  
   - Transformed extracted features to meet required dimensions for downstream tasks:
     ```python
     full_features = full_features[:,:,:,0,0,0]
     full_features = np.array(full_features).transpose([1, 0, 2])
     ```

5. **Improved Logging**  
   - Added logs for parameters like `batch_size` to improve runtime visibility:
     ```python
     print("batchsize", batch_size)
     ```

---

### Changes to `main.py`

The `main.py` script was enhanced for better user experience and feature flexibility. Key updates include:

1. **Summary Logging**  
   - Logs details of each run in `feature_config.txt`, including:
     - Run date/time, total videos, skipped videos, and processed videos.
     ```json
     {
         "run_date": "2024-12-02 12:00:00",
         "total_videos": 100,
         "videos_skipped": 5,
         "processed_videos": 95
     }
     ```

2. **Graceful Skipping of Videos**  
   - Skipped videos due to insufficient frames are logged:
     ```plaintext
     Feature extraction skipped for video_name due to insufficient frames.
     ```

3. **Feature Metadata Collection**  
   - Metadata for extracted features (video name, file path, dimensions) is logged in `feature_config.txt`.

4. **Temporary Directory Management**  
   - Ensured temporary directories (`temp/`) are cleaned up after processing to prevent residual files.

5. **Flexible Output Paths**  
   - Automatically creates nested directories for output if they do not exist.

6. **Enhanced Logs**  
   - Added detailed logs for video processing status and total processing time:
     ```plaintext
     Processing video 1: /path/to/video.mp4
     Preprocessing done..
     Features extracted and saved to /output_features/video1.npy with shape (20, 10, 1024)
     Processing time: 3.25s
     ```

---

These updates enhance the scripts’ usability, efficiency, and compatibility with diverse datasets.
## How to Run
1. **Install Dependencies**
   - Python 3.11.4
   - Install libraries using `pip install -r requirements.txt`

2. **Prepare Your Data**
   - Dataset structure:
     ```
     /dataset_root/
         train/
         val/
         test/
     ```
   - Any preprocessing steps.

3. **Run the Code**
   ```bash
   python main.py --datasetpath=sample
   videos/ --outputpath=output
   ```



