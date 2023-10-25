#!/usr/bin/env bash

# Crop images from megadetector results
python3 crop_detections.py \
   "../../missouri_data_processed/missouri-camera-traps_mdv5b.0.0_results.json" \
   "../../missouri_data_processed/md_cropped" \
   --images-dir "../../missouri_data_raw/" \
   --threshold 0.85 \
   --square-crops \
   --logdir "."

## BROKEN: Create and save TF datasets from the crops
# python3 create_dataset.py \
#    "../../missouri_data_processed/md_cropped/Set1" \
#    "../../missouri_data_processed/tf_datasets" \
#    32 \
#    224 \
#    224 \
#    0.2
