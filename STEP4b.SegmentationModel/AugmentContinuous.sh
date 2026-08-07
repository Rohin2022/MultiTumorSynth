#!/bin/bash

while true; do
    python AugmentExternal.py --dataset atlas --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --workers_overwrite 4 --save_destination /scratch/rpinise1/MultiTumorSynthesis/SegTrain_AugCombinedV2 --dataset_path /scratch/rpinise1/MultiTumorSynthesis/Synthetic_RSuperProcessed_VV2_COMBINED_NPZ
    echo "Script exited with code $?; restarting in 10 seconds..."
    sleep 10
done