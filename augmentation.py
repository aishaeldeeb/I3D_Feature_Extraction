import cv2
import numpy as np
import os

# Define data augmentation functions
def brighten(img, level):
    img = np.clip(img * level, 0, 255).astype(np.uint8)
    return img

# Define paths and read video directories from list
base_path = "/home/aishaeld/scratch/I3D_Feature_Extraction_resnet/"
list_path = "/home/aishaeld/scratch/RTFM/list/train_list_v2.list"
data_path = os.path.join(base_path, "data_v2")
augmented_path = os.path.join(base_path, "augmented_data")
count = 0

with open(list_path, "r") as f:
    video_dirs = f.readlines()

# Loop through each video directory and perform data augmentation
for video_dir in video_dirs:
    count+=1
    print(f'video number: {count}')
    print(video_dir)

    directory = os.path.basename(os.path.dirname(video_dir))
    desired_directory_name = directory.split("_")[0] 
    print(desired_directory_name)
    if(desired_directory_name == 'non'):
        desired_directory_name = 'non_anomaly'

    video_dir = video_dir.strip()
    video_name = os.path.basename(video_dir)
    video_name = os.path.splitext(video_name)[0] + ".mp4"
    
    video_path = os.path.join(data_path, video_dir.replace("features_v2", "data_v2").replace(".npy", ".mp4"))
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print('3 levels of brightness augmentation')
    # Perform 3 levels of brightness augmentation and save each augmented video
    for j in range(1, 4):
        brightened_path = os.path.join(augmented_path, desired_directory_name,  f"{os.path.splitext(video_name)[0]}_bright{j}.mp4")
        writer = cv2.VideoWriter(brightened_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while True:
            ret, frame = video.read()
            if not ret:
                break
            brightened = brighten(frame, j)
            writer.write(brightened)

        writer.release()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print('horizontal flip')
    # Perform horizontal and vertical flip augmentation and save each augmented video
    flipped_h_path = os.path.join(augmented_path, desired_directory_name, f"{os.path.splitext(video_name)[0]}_flip_h.mp4")
    writer = cv2.VideoWriter(flipped_h_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        flipped_h = cv2.flip(frame, 1)
        writer.write(flipped_h)

    writer.release()
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print('vertical flip')

    flipped_v_path = os.path.join(augmented_path, desired_directory_name,  f"{os.path.splitext(video_name)[0]}_flip_v.mp4")
    writer = cv2.VideoWriter(flipped_v_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        flipped_v = cv2.flip(frame, 0)
        writer.write(flipped_v)

    writer.release()
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Release the video capture object
    video.release()