# conda create -n opensora python=3.9
# conda activate opensora
pip install torch==2.5.1  --index-url https://download.pytorch.org/whl/cu124
pip install torchvision+cu124  --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.25.post1  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt # for development mode, `pip install -v -e .`
pip install -v . # for development mode, `pip install -v -e .`
pip install git+https://github.com/hpcaitech/TensorNVMe
pip install git+https://github.com/hpcaitech/ColossalAI
# install flash attention
# set enable_flash_attn=False in config to disable flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex
# set enable_layernorm_kernel=False in config to disable apex
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git