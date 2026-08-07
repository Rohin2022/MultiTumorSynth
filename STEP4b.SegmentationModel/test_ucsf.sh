#!/bin/bash


# Specify which GPUs to use (adjust as needed)
gpus=(0)
part_list=(0)
total_parts=1

# Path to the test set
test_set='/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/'

# Retry controls (override via env vars if you want)
max_retries=25000
retry_sleep="${RETRY_SLEEP:-20}"

models=(
    #"./exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_attenuation_slice_loss_all_data_lr4_RSuperMTL_cls_on_segmentation_att_classifier_venous_only_no_mask_ft_slices_only_binarized_cls/fold_0_latest.pth"
    #"./exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_attenuation_slice_loss_all_data_lr4_RSuperMTL_cls_on_segmentation_att_classifier_venous_only_no_mask_ft_slices_only/fold_0_latest.pth"
    #"./exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_attenuation_slice_loss_all_data_lr4_RSuperMTL_cls_on_segmentation_att_classifier_venous_only/fold_0_latest.pth"
    
    #"exp/abdomenatlas/mask_only_model_name/fold_0_latest.pth"
    #"/projects/bodymaps/Pedro/foundational/MedFormer/exp/abdomenatlas/PRETRAIN_UCSF_133K_and_Merlin_w0_many_cancers_100_epch/fold_0_latest.pth" # ORIGINAL MODEL
    #"/projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/exp/abdomenatlas/mask_only_model_name/fold_0_epoch_25.pth"
    #"/projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/exp/abdomenatlas/mask_only_combined_data/fold_0_epoch_1.pth"
    #"/projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/exp/abdomenatlas/mask_only_combined_data/fold_0_epoch_1.pth"
    #"./exp/abdomenatlas/clip_ft_UCSF133K/fold_0_latest.pth"
    #"/projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/exp/abdomenatlas/v1_mask_only_synth_and_real_1_to_1/fold_0_epoch_11.pth"
    #"/projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/exp/abdomenatlas/v1_mask_only_synth_and_real_1_to_1/fold_0_epoch_16.pth"
    #"/projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/exp/abdomenatlas/v1_mask_only_synth_and_real_1_to_1/fold_0_latest.pth"
    "/projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/exp/abdomenatlas/v2_mask_only_synth_and_real_1_to_1/fold_0_latest.pth"


    #"./exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_attenuation_slice_loss_all_data_lr4_RSuperMTL_cls_on_segmentation_att_classifier_venous_only_no_mask/fold_0_latest.pth"
    #"./exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_attenuation_slice_loss_all_data_lr4_RSuperMTL_cls_on_segmentation_att_classifier_venous_only_50_masks/fold_0_latest.pth"
    #"./exp/abdomenatlas/PRETRAIN_UCSF_133K_and_Merlin_w0_many_cancers_100_epch_50_masks/fold_0_latest.pth"
    #"./exp/abdomenatlas/PRETRAIN_UCSF_133K_and_Merlin_w0_many_cancers_100_epch_25_masks/fold_0_latest.pth"
    #"./exp/abdomenatlas/genesis_ft_UCSF133K/fold_0_latest.pth"
    #"./exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_all_data_MTL/fold_0_latest.pth"
    #"./exp/abdomenatlas_ufo_multi_tumor/Dataset_133K_merlin_ucsf_attenuation_slice_loss_all_data_lr4_RSuperMTL_cls_on_segmentation_att_classifier_venous_only_25_masks/fold_0_latest.pth"
    )


# Function to run a model on a specified GPU and part
run_model() {
    local model=$1
    local test_set=$2
    local gpu=$3
    local parts=$4
    local current_part=$5
    local max_retries=$6
    local retry_sleep=$7

    echo "Running model ${model} on GPU: ${gpu} (part ${current_part}/${parts})"

    local extra_flag=""
    if [[ "${model}" == *"gate"* ]]; then
        extra_flag="--classification_branch --cls_gate"
    fi
    
    if [[ "${model}" == *"Y_net"* || "${model}" == *"classification_baseline"* ]]; then
        extra_flag="--classification_branch"
    fi

    if [[ "${model}" == *"cls_on_segmentation"* ]]; then
        extra_flag="--cls_on_segmentation"
    fi

    if [[ "${model}" == *"malignancy"* ]]; then
        extra_flag+=" --malignancy_classification"
    fi

    if [[ "${model}" == *"binarize"* ]]; then
        extra_flag+=" --binarize_cls_on_segmentation"
    fi

    local attempt=1
    while [[ "${attempt}" -le "${max_retries}" ]]; do
        python predict_abdomenatlas.py --load "${model}" \
            --img_path "${test_set}" \
            --class_list /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/preprocessing/label_names.yaml \
            --gpu ${gpu} --organ_mask_on_lesion ${extra_flag} \
            --save_path ./result/v2_epoch_final/ \
            --parts ${total_parts} --current_part ${current_part} \
            --ids /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/cross_eval/testing_ids_turkish.csv \
            --meta /projects/bodymaps/Data/metadata_ucsf_batch_1_to_6_and_merlin.csv \
            --reports /projects/bodymaps/Data/metadata_per_tumor_ucsf_batch_1_to_6_and_merlin.csv \
            --disable_inference_2_stages
            #--overwrite \
            #--ids /projects/bodymaps/Data/Merlin/merlin_8tumor_cases_and_normal.csv

             # use  /projects/bodymaps/Pedro/data/UCSF_merlin_Sep16_radiologist_annotations_medformer_npz/list/label_names.yaml for regular model
             # use /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/preprocessing/label_names.yaml for finetuned model

        rc=$?
        if [[ "${rc}" -eq 0 ]]; then
            echo "Model ${model} executed successfully on GPU ${gpu} (part ${current_part})."
            return 0
        fi

        if [[ "${attempt}" -ge "${max_retries}" ]]; then
            echo "ERROR: Model ${model} failed on GPU ${gpu} (part ${current_part}) after ${max_retries} attempts. Giving up."
            return "${rc}"
        fi

        echo "Model ${model} failed on GPU ${gpu} (part ${current_part}). Attempt ${attempt}/${max_retries}. Retrying in ${retry_sleep}s..."
        sleep "${retry_sleep}"
        attempt=$((attempt + 1))
    done
}

# Trap to handle Ctrl+C and stop all running processes
trap "echo 'Stopping all processes...'; kill 0; exit 1" SIGINT

# Process each model sequentially, but run its parts concurrently
for model in "${models[@]}"; do
    echo "Processing model: $model"

    pids=()

    # iterate over indices of part_list
    for idx in "${!part_list[@]}"; do
        part_real="${part_list[$idx]}"                     # actual part number
        gpu="${gpus[$(( idx % ${#gpus[@]} ))]}"            # round-robin GPU

        run_model "$model" "$test_set" "$gpu" \
                  "$total_parts" "$part_real" \
                  "$max_retries" "$retry_sleep" &
        pids+=($!)
    done

    # Wait for all parts; fail fast if any part fails (e.g., hit max_retries)
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done

    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: At least one part failed for model: $model"
        exit 1
    fi
done

echo "All models processed."