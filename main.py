from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
from resnet import i3_res50
import os
from datetime import datetime
import json

def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = outputpath + "/temp/"
    rootdir = Path(datasetpath)
    videos = [str(f) for f in rootdir.glob('**/*.mp4')]

    # Setup the model
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    count = 0
    skipped_videos = []
    extracted_features = []

    for video in videos:
        count += 1
        videoname = video.split("/")[-1].split(".")[0]
        startime = time.time()
        print(f"Processing video {count}: {video}")

        # Create temporary directory for extracted frames
        Path(temppath).mkdir(parents=True, exist_ok=True)
        ffmpeg.input(video).output('{}%d.jpg'.format(temppath), start_number=0).global_args('-loglevel', 'quiet').run()
        print("Preprocessing done..")

        # Run feature extraction
        features = run(i3d, frequency, temppath, batch_size, sample_mode)

        if features is None:
            print(f"Feature extraction skipped for {videoname} due to insufficient frames.")
            skipped_videos.append(video)
        else:
            npy_file = os.path.join(outputpath, f"{videoname}.npy")
            np.save(npy_file, features)
            extracted_features.append({
                "video_name": videoname,
                "feature_file": npy_file,
                "feature_size": features.shape
            })
            print(f"Features extracted and saved to {npy_file} with shape {features.shape}")

        # Clean up temporary directory
        shutil.rmtree(temppath)
        print(f"Processing time: {time.time() - startime:.2f}s")

    # Prepare summary for this run
    current_run_summary = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(videos),
        "videos_skipped": len(skipped_videos),
        "skipped_video_list": skipped_videos,
        "processed_videos": len(extracted_features)
    }

    config_file_path = os.path.join(outputpath, "feature_config.txt")

    # Append the current run summary
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as config_file:
            existing_data = json.load(config_file)
    else:
        existing_data = []

    existing_data.append(current_run_summary)

    with open(config_file_path, "w") as config_file:
        config_file.write(json.dumps(existing_data, indent=4))

    print(f"Feature extraction completed. Summary updated in {config_file_path}")

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="samplevideos/")
	parser.add_argument('--outputpath', type=str, default="output_features")
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)    
