# Define dialects and modes
DIALECTS=("aae" "bre" "che" "ine" "sge")
MODES=("concise" "detailed")
BASE_MASTER_PORT=29508
BASE_RDZV_PORT=30508

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
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc-per-node=1 \
        --master-port=$CURRENT_MASTER_PORT \
        --rdzv-endpoint=localhost:$CURRENT_RDZV_PORT \
        cosmos1/models/diffusion/inference/text2world_multi1.py \
        --dialect $dialect \
        --mode $mode \
        --checkpoint_dir '/home/weights/checkpoints' \
        --offload_prompt_upsampler --offload_diffusion_transformer

    # Increment ports for next run
    BASE_MASTER_PORT=$((BASE_MASTER_PORT + 1))
    BASE_RDZV_PORT=$((BASE_RDZV_PORT + 1))
    
    echo "✅ Completed ${dialect} (${mode} mode)"
    echo ""
  done
done

echo "🎉 All dialects and modes completed successfully!"