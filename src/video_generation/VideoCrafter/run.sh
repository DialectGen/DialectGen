#!/bin/bash

# Define dialects and modes
DIALECTS=("aae" "bre" "che" "ine" "sge")
MODES=("concise" "detailed")
BASE_MASTER_PORT=29500
BASE_RDZV_PORT=30500

for dialect in "${DIALECTS[@]}"; do
  for mode in "${MODES[@]}"; do
    # Calculate ports for this run
    CURRENT_MASTER_PORT=$BASE_MASTER_PORT
    CURRENT_RDZV_PORT=$BASE_RDZV_PORT
    
    echo "=================================================="
    echo "🚀 Starting generation for ${dialect} (${mode} mode)"
    echo "📡 Using ports: master=${CURRENT_MASTER_PORT}, rdzv=${CURRENT_RDZV_PORT}"
    echo "=================================================="
    
    # Run with waiting
    CUDA_VISIBLE_DEVICES=1,2 torchrun \
        --nproc-per-node=2 \
        --master-port=$CURRENT_MASTER_PORT \
        --rdzv-endpoint=localhost:$CURRENT_RDZV_PORT \
        scripts/evaluation/inference.py \
        --dialect $dialect \
        --mode $mode \
        --config 'configs/inference_t2v_512_v2.0.yaml' \
        --ckpt '/local2/cipeng/weights/VideoCrafter2/model.ckpt' \
        --guidance_scale 12.0 \
        --ddim_steps 50 \
        --ddim_eta 1.0 \
        --overwrite

    # Increment ports for next run
    BASE_MASTER_PORT=$((BASE_MASTER_PORT + 1))
    BASE_RDZV_PORT=$((BASE_RDZV_PORT + 1))
    
    echo "✅ Completed ${dialect} (${mode} mode)"
    echo ""
  done
done

echo "🎉 All dialects and modes completed successfully!"