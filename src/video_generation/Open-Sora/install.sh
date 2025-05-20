pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torchvision -U  --index-url https://download.pytorch.org/whl/cu126
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt # for development mode, `pip install -v -e .`
pip install -v . # for development mode, `pip install -v -e .`
pip install git+https://github.com/hpcaitech/TensorNVMe
pip install git+https://github.com/hpcaitech/ColossalAI
pip install packaging ninja
pip install flash-attn --no-build-isolation
