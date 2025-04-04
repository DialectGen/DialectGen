#!/bin/bash

# Define dialects and modes
DIALECTS=("aae" "bre" "che" "ine" "sge")
MODES=("concise" "detailed")
BASE_MASTER_PORT=29507
BASE_RDZV_PORT=30507

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
    CUDA_VISIBLE_DEVICES=6,7 torchrun \
        --nproc-per-node=2 \
        --master-port=$CURRENT_MASTER_PORT \
        --rdzv-endpoint=localhost:$CURRENT_RDZV_PORT \
        cogvideox5.py \
        --dialect $dialect \
        --mode $mode \
        --overwrite

    # Increment ports for next run
    BASE_MASTER_PORT=$((BASE_MASTER_PORT + 1))
    BASE_RDZV_PORT=$((BASE_RDZV_PORT + 1))
    
    echo "✅ Completed ${dialect} (${mode} mode)"
    echo ""
  done
done

echo "🎉 All dialects and modes completed successfully!"