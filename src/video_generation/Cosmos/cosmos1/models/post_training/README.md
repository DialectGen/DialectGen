# Cosmos Post-training

In the [Cosmos paper](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai), we discuss several post-training examples of Cosmos pre-trained World Foundation Models (WFMs) for various Physical AI tasks, including

- instruction control
- camera control
- action control
- multi-view generation
- multi-view generation with vehicle trajectory control

Except for the instruction control where the WFM is fine-tuned on a dataset of instruction-video pairs, all other cases require minor modifications of the network architectures. In this initial release, we provide post-training scripts for the instruction control fine-tuning for both diffusion- and the autorgressive-based WFMs. Scripts of the other post-training examples will be provided in a future release.

| Post-training Task  | Diffusion WFM | Autoregressive WFM |
|---------------------|---------------|--------------------|
| Instruction control | ETA: 01/07/2025 | ETA: 01/07/2025 |
| Action control | Coming soon | Coming soon |
| Camera control | Coming soon | Coming soon |
| Multi-view generation | Coming soon | Coming soon |
| Multi-view generation with vehicle trajectory control | Coming soon | Coming soon |
