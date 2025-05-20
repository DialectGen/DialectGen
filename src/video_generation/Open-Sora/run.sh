#!/bin/bash

# Define dialects and modes
DIALECTS=("aae" "bre" "che" "ine" "sge")
MODES=("concise" "detailed")
BASE_MASTER_PORT=29908
BASE_RDZV_PORT=30178

for dialect in "${DIALECTS[@]}"; do
  for mode in "${MODES[@]}"; do
    CURRENT_MASTER_PORT=$BASE_MASTER_PORT
    CURRENT_RDZV_PORT=$BASE_RDZV_PORT
    
    echo "=================================================="
    echo "🚀 Starting generation for ${dialect} (${mode} mode)"
    echo "📡 Using ports: master=${CURRENT_MASTER_PORT}, rdzv=${CURRENT_RDZV_PORT}"
    echo "=================================================="
    
    # Run with waiting

    sed -i "s/^dialect *=.*/dialect = '$dialect'/" config.py
    sed -i "s/^mode *=.*/mode = '$mode'/" config.py
    
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=$CURRENT_MASTER_PORT scripts/inference_multi.py /local/cipeng/multimodal-dialectal-bias/src/video_generation/Open-Sora/config.py \
       --num-frames 51 --resolution 720p --aspect-ratio 9:16 --batch-size 1

    # Increment ports for next run
    BASE_MASTER_PORT=$((BASE_MASTER_PORT + 1))
    BASE_RDZV_PORT=$((BASE_RDZV_PORT + 1))
    
    echo "✅ Completed ${dialect} (${mode} mode)"
    echo ""
  done
done

echo "🎉 All dialects and modes completed successfully!"