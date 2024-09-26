#!/bin/sh

# Array of source directories
source_dirs=(
  final_data_v2/train_val/non_anomaly/*
  final_data_v2/train_val/non_anomaly_cropped/*
  final_data_v2/train_val/non_anomaly_augmented/*
  final_data_v2/test/anomaly/*
  final_data_v2/test/non_anomaly/*
)

# Corresponding array of destination directories
dest_dirs=(
  final_augmented_data/train_val/non_anomaly/
  final_augmented_data/train_val/non_anomaly/
  final_augmented_data/train_val/non_anomaly/
  final_augmented_data/test/anomaly/
  final_augmented_data/test/non_anomaly/
)

# Ensure that the number of source and destination directories are the same
if [ ${#source_dirs[@]} -ne ${#dest_dirs[@]} ]; then
  echo "Error: The number of source and destination directories must be the same."
  exit 1
fi

# Loop over the directories and copy files
for i in ${!source_dirs[@]}; do
  src="${source_dirs[$i]}"
  dest="${dest_dirs[$i]}"

  # Create the destination directory if it does not exist
  mkdir -p "$dest"

  # Copy files
  cp -r "$src" "$dest"
done

echo "File copy completed successfully."
