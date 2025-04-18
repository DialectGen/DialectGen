#!/bin/bash

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
    export PYTHONPATH=/local/cipeng/multimodal-dialectal-bias/src/video_generation/Cosmos:$PYTHONPATH
    CUDA_VISIBLE_DEVICES=3 torchrun \
        --nproc-per-node=1 \
        --master-port=$CURRENT_MASTER_PORT \
        --rdzv-endpoint=localhost:$CURRENT_RDZV_PORT \
        cosmos1/models/diffusion/inference/text2world_multi1.py \
        --dialect $dialect \
        --mode $mode \
        --checkpoint_dir '/local2/cipeng/weights/checkpoints' \
        --offload_prompt_upsampler --offload_diffusion_transformer
        # --offload_diffusion_transformer --offload_tokenizer --offload_text_encoder_model --offload_prompt_upsampler --offload_guardrail_models
        # --config 'configs/inference_t2v_512_v2.0.yaml'  


    # parser.add_argument("--dialect", required=True, choices=list(ENTIGEN_PREFIXES.keys()))
    # parser.add_argument("--mode", required=True, choices=["concise", "detailed", "entigen", "polysemy"])
    # parser.add_argument("--overwrite", action="store_true")
    # parser.add_argument("--diffusion_transformer_dir", type=str, default="Cosmos-1.0-Diffusion-7B-Text2World")
    # parser.add_argument("--prompt_upsampler_dir", type=str, default="Cosmos-1.0-Prompt-Upsampler-12B-Text2World")
    # parser.add_argument("--word_limit_to_skip_upsampler", type=int, default=250)

    # Increment ports for next run
    BASE_MASTER_PORT=$((BASE_MASTER_PORT + 1))
    BASE_RDZV_PORT=$((BASE_RDZV_PORT + 1))
    
    echo "✅ Completed ${dialect} (${mode} mode)"
    echo ""
  done
done

echo "🎉 All dialects and modes completed successfully!"