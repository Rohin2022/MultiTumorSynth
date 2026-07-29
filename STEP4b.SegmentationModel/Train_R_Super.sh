#!/bin/bash

# Trap Ctrl+C and exit immediately
trap "echo 'Ctrl-C detected. Exiting...'; exit 1" SIGINT

#while true; do
    python train_ddp.py --dataset abdomenatlas_ufo_multi_tumor --model medformer --dimension 3d --batch_size 2 --unique_name TumorSynth_Seg_V1 \
    --crop_on_tumor --gpu '0' --workers 4 --classes_number 55 \
    --pretrain --pretrained /projects/bodymaps/Pedro/foundational/MedFormer/exp/abdomenatlas/PRETRAIN_UCSF_133K_and_Merlin_w0_many_cancers_100_epch/fold_0_latest.pth \ #pretrained weights
    --loss ball_dice_last --dist_url tcp://127.0.0.1:9744 --report_volume_loss_basic 0.1 \
    --save_destination /projects/bodymaps/Pedro/data/AbdomenAtlasRadiologist_multi_tumor_UCSF_batch_1_to_5_and_Merlin_MedformerNpzAugmentedBalancedCropperWithSliceAllData/ \
    --data_root /projects/bodymaps/Pedro/data/UCSF_merlin_Sep16_radiologist_annotations_medformer_npz/ \
    --UFO_root /projects/bodymaps/Data/UCSF_batch_1_to_5_and_merlin_medformer_npz_symlinks/ \
    --ucsf_ids /projects/bodymaps/Data/UCSF_133K_Train_Set.csv \
    --reports /projects/bodymaps/Data/metadata_per_tumor_ucsf_batch_1_to_6_and_merlin.csv \
    --test_ids_exclude /projects/bodymaps/Data/UCSF_133K_Test_Set.csv \
    --epochs 30 --lr 0.0001 --balanced_cropper --attenuation_classifier 'MLP' --attenuation_classifier_venous \
    --att_weight 0.01 \
    --slice_loss --load_augmented --use_all_data --use_sample_weigths \
    --cls_on_segmentation \
    #--resume --load /projects/bodymaps/Pedro/foundational/MedFormer/exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_attenuation_slice_loss_all_data_lr4_RSuperMTL_cls_on_segmentation_att_classifier_venous_only/fold_0_latest.pth


    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        break
    else
        echo "Error encountered (exit code $exit_code). Restarting in 20 seconds..."
        sleep 20
    fi
#done


