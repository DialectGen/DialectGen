
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/inference.py configs/opensora-v1-2/inference/sample.py --num-frames 4s --resolution 720p --aspect-ratio 9:16 --batch-size 2

CUDA_VISIBLE_DEVICES=4,5 python scripts/inference_multi.py configs/opensora-v1-2/inference/sample.py --num-frames 1s --resolution 720p --aspect-ratio 9:16 --batch-size 1
