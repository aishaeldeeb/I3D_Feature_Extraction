import os

file_path = "/scratch/aishaeld/I3D_Feature_Extraction_resnet/features_v2/anomaly/StayLong5.npy"
directory = os.path.basename(os.path.dirname(file_path))
desired_directory_name = directory.split("_")[0]  # Assuming "anomaly" is the first part of the directory name
print(desired_directory_name)