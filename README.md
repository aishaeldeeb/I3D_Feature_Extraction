# Customized I3D Feature Extraction with ResNet

## Overview
This repository is a customized version of [Original Repo](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet). The changes were made to better suit the needs of [Your Dataset or Task]. It extracts video features efficiently using ResNet.

## Changes Made
- **Feature X**: Description of the feature, why it was added.
- **Bug Fixes**: List any fixes.
- **Other Adjustments**: Briefly explain changes to structure or code.

## How to Run
1. **Install Dependencies**
   - Python 3.X
   - Libraries: `torch`, `numpy`, `torchvision`...
   - Install using `pip install -r requirements.txt`

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
   python main.py --datasetpath=samplevideos/ --outputpath=output
   ```

# Install ffmpeg (static binary)
FFMPEG_DIR=$TMPDIR/ffmpeg
mkdir -p $FFMPEG_DIR
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O $FFMPEG_DIR/ffmpeg.tar.xz
tar -xf $FFMPEG_DIR/ffmpeg.tar.xz -C $FFMPEG_DIR --strip-components=1
export PATH=$FFMPEG_DIR:$PATH

Verify ffmpeg installation
ffmpeg -version

