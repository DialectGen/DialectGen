export CUDA_VISIBLE_DEVICES=0,1,2,3

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
    torchrun --nproc_per_node=4 --master_port=29505 generate_multi.py \
        --dialect $dialect \
        --mode $mode \
        --task t2v-14B \
        --size 832*480 \
        --frame_num 10 \
        --ckpt_dir /local2/cipeng/weights/Wan2.1-T2V-14B \
        --dit_fsdp \
        --t5_fsdp \
        --ring_size 4 \
        --ulysses_size 1 \
        --sample_steps 12 \
        --sample_shift 5.0 \
        --sample_guide_scale 5.0 \
        --base_seed 42 \
        --offload_model true

    # Increment ports for next run
    BASE_MASTER_PORT=$((BASE_MASTER_PORT + 1))
    BASE_RDZV_PORT=$((BASE_RDZV_PORT + 1))
    
    echo "✅ Completed ${dialect} (${mode} mode)"
    echo ""
  done
done

echo "🎉 All dialects and modes completed successfully!"