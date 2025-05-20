# Video Generation Model Setup Guide

## Installation

We **strongly recommend** creating a fresh Conda environment per model to avoid dependency conflicts.

1. **Create & activate** environment  
   ```bash
   conda create -n <env_name> python=3.10 -y
   conda activate <env_name>

2. **Install model-specific requirements**
``` pip install -r <ModelDir>/requirements.txt ```

3. **Install CogVideo dependencies**
``` pip install diffusers accelerate transformers ```

## Usage
1. **Running a Directory Model**

Each subfolder ships a run.sh. 

Example for VideoCrafter:

```bash
cd VideoCrafter
bash run.sh
```

2. **Running CogVideoX5B**

CogVideo is implemented in diffusers, so you can run it if you have the diffusers library installed.

To run cogvideo, use simply run the `gen_cog.sh` bash file.
```bash
bash gen_cog.sh
```