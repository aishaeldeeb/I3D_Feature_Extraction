from VideoProcessor import VideoProcessor


if __name__ == "__main__":
    # List of directories to process
    directories = [
        "/home/aishaeld/scratch/I3D_Feature_Extraction_resnet/final_data_v2/train_val/anomaly",
        "/home/aishaeld/scratch/I3D_Feature_Extraction_resnet/final_data_v2/train_val/non_anomaly",
        "/home/aishaeld/scratch/I3D_Feature_Extraction_resnet/final_data_v2/test/anomaly",
        "/home/aishaeld/scratch/I3D_Feature_Extraction_resnet/final_data_v2/test/non_anomaly"
    ]

    # Loop over the directories and process videos
    for dir in directories:
        processor = VideoProcessor(dir, dir + "_augmented")
        processor.process_videos()


