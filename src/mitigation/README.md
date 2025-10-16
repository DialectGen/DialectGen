## Setup
Navigate to the directory by running the following command:
```
cd src/mitigation
```
You need to install the following additional packages in the Conda environment that you created from `environment.yml`.
```
pip install wandb
pip install datasets==3.1.0
```


## Download MS COCO dataset
First, if the MSCOCO dataset is not available, you will need to download it. Please refer to the `download_mscoco.sh` for instructions. This will create a folder named `mscoco` under the `data` directory and download the data into it.
```
bash download_mscoco.sh
```


## Fine-tune the text encoder
You can fine-tune the text encoder using the following command. The relevant configuration is included in `configs` folder.
```
CUDA_VISIBLE_DEVICES=[GPU] python finetune.py --config configs/sd15.yaml
```


## Generate images
After the encoder has been fine-tuned, images are generated using the fine-tuned encoder. You should specify the path to the fine-tuned encoder in `ENCODER_PATH`. If `SWAP=1`, the image is generated using the fine-tuned encoder (i.e., swapped in). If `SWAP=0`, the original encoder is used for image generation.

### Diaelct/SAE
```
CUDA_VISIBLE_DEVICES=[GPU] python generate_images.py --model stable-diffusion-v1-5/stable-diffusion-v1-5 --encoder [ENCODER_PATH] --swap [SWAP] --dialect [DIALECT]
```

### SAE Polysemy
```
CUDA_VISIBLE_DEVICES=[GPU] python generate_images_polysemy.py --model stable-diffusion-v1-5/stable-diffusion-v1-5 --encoder [ENCODER_PATH] --swap [SWAP] --dialect [DIALECT]
```

### SAE MSCOCO
```
CUDA_VISIBLE_DEVICES=[GPU] python generate_images_mscoco.py --model $model --encoder [ENCODER_PATH] --swap [SWAP]
```


## Evaluation using VQA
Once all images are generated, we perform scoring using the VQA metric. The `RES_DIR` argument is the directory where the images are generated.

### Dialect/SAE
```
CUDA_VISIBLE_DEVICES=[GPU] python vqa_score_understanding.py --res_dir [RES_DIR] --dialect [DIALECT]
```

### SAE Polysemy
```
CUDA_VISIBLE_DEVICES=[GPU] python vqa_score_understanding_polysemy.py --res_dir [RES_DIR] --dialect [DIALECT]
```
Then a file named `vqa_score_understanding_polysemy.json` will be created under the `RES_DIR` directory. Run the following script to aggregate the results.
```
python aggregate_polysemy_res.py --res_path [RES_PATH]
```
`RES_PATH` refers to the absolute path of `vqa_score_understanding_polysemy.json`.

### SAE MSCOCO
```
CUDA_VISIBLE_DEVICES=[GPU] python vqa_score_understanding_mscoco.py --res_dir [RES_DIR]
```