export EXPERIMENT_NAME="sd-naruto-model"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
export TRAIN_FILE_NAME="/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/diffusers/examples/text_to_image/train_text_to_image.py"
export CHECKPOINTS_FOLDER="/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/checkpoints"

accelerate launch --mixed_precision="fp16" $TRAIN_FILE_NAME \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --use_ema \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=15000 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --validation_prompt="cute dragon creature" --report_to="wandb" \
    --output_dir="${CHECKPOINTS_FOLDER}/${EXPERIMENT_NAME}"