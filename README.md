<h1 align="center">DialectGen: Benchmarking and Improving Dialect Robustness in Multimodal Generation</h1>
<p align="center">
    <a>Under Submission</a> 
</p>


<p align="center">
    <a href="https://arxiv.org/">
        <img src="https://img.shields.io/badge/arXiv-2410.07166-B31B1B.svg?style=plastic&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://dialectgen.github.io/">
        <img src="https://img.shields.io/badge/Website-DialectGen-purple?style=plastic&logo=Google%20chrome" alt="Website">
    </a>
    <a href="https://huggingface.co/datasets/uclanlp/" target="_blank">
        <img src="https://img.shields.io/badge/Dataset-Download-yellow?style=plastic&logo=huggingface" alt="Download the EmbodiedAgentInterface Dataset from Hugging Face">
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=plastic" alt="License: MIT">
    </a>

</p>

<p align="center">
    <a href="https://yu-bryan-zhou.github.io/">Yu Zhou*†</a>, 
    <a href="https://cownowan.github.io/">Sohyun An*</a>, 
    <a href="https://haikangdeng.github.io/">Haikang Deng*</a>, 
    <a href="https://wadeyin9712.github.io/">Da Yin</a>, 
    <a href="https://clarkipeng.github.io/">Clark Peng</a>, <br>
    <a href="https://web.cs.ucla.edu/~chohsieh/">Cho-Jui Hsieh</a>,
    <a href="https://web.cs.ucla.edu/~kwchang/">Kai-Wei Chang</a>, 
    <a href="https://violetpeng.github.io/">Nanyun Peng</a>
</p>


<p align="center">
    <a href="https://dialectgen.github.io/" target="_blank">
        <img src="./assets/method.png" alt="CoDA" width="90%" height="90%" border="10" />
    </a>
</p>


## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Bibtex](#acknowledgements)

---

# Introduction

Contact languages like English exhibit rich regional variations in the form of dialects, which are often used by dialect speakers interacting with generative models. However, can multimodal generative models effectively produce content given dialectal textual input? In this work, we study this question by constructing a new large-scale benchmark spanning six common English dialects. We work with dialect speakers to collect and verify over 4200 unique prompts and evaluate on 17 image and video generative models. Our automatic and human evaluation results show that current state-of-the-art multimodal generative models exhibit 32.26% to 48.17% performance degradation when a single dialect word is used in the prompt. Common mitigation methods such as fine-tuning and prompt rewriting can only improve dialect performance by small margins (< 7%), while potentially incurring significant performance degradation in Standard American English (SAE). To this end, we design a general encoder-based mitigation strategy for multimodal generative models. Our method teaches the model to recognize new dialect features while preserving SAE performance. Experiments on models such as Stable Diffusion 1.5 show that our method is able to simultaneously raise performance on five dialects to be on par with SAE (+34.4%), while incurring near zero cost to SAE performance.

# Quick Start

## 1. Installation

Build the DialectGen environment from conda:

```bash
   conda env create -f environment.yml
   conda activate DialectGen
```


## 2. Evaluate Image or Video Generative Models on DialectGen

### 1.1 Image Generation

For existing models in the DialectGen paper, please use scripts in `src/img_generation`. To evaluate your own model, please duplicate any existing script in `src/img_generation` and modify with your model generation function.

```bash
python src/img_generation/sd35-turbo.py --dialects aae bre che ine sge --mode concise
```

##### Arguments

- `--dialects`: The dialects you want to generate images for, can be any of: [aae, bre, che, ine, sge].
- `--mode`: Which evaluation mode you would like to use, can be any of: [concise, detailed, polysemy].
- `--replace`: Add this argument if you would like to re-generate images for the given dialect and mode.


#### Output Structure

```
DialectGen/
├── data/
│   └── image/
│       └── {mode}/
│           └── {model}/
│               ├── sae_images/
│                   └── ...
│               └── dialect_imgs/
│                   └── {prompt}
│                       ├── 0.jpg
│                       ├── ...
│                       ├── 9.jpg
```

### 1.2 Video Generation


#### Installation

We **strongly recommend** creating a fresh Conda environment per model to avoid dependency conflicts.

##### **Create & activate** environment  

```bash
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
```

##### **Install model-specific requirements**
```bash 
pip install -r src/video_generation/<ModelDir>/requirements.txt 
```

##### **Install CogVideo dependencies**
```bash
pip install diffusers accelerate transformers 
```

#### Usage
##### **Running a Directory Model**

Each subfolder ships a run.sh. 

Example for VideoCrafter:

```bash
cd src/video_generation/VideoCrafter
bash run.sh
```

##### **Running CogVideoX5B**

CogVideo is implemented in diffusers, so you can run it if you have the diffusers library installed.

To run cogvideo, use simply run the `gen_cog.sh` bash file.

```bash
bash gen_cog.sh
```


### 1.3 Evaluation

#### Installation

Please follow instructions in the [VQAScore Github Repo](https://github.com/linzhiqiu/t2v_metrics) to create the conda environment `t2v`.

```bash
conda activate t2v
```

#### VQA Score and CLIP Score Evaluation

Run the following scripts with the required parameters:

For VQA Score evaluation:
```shell
python src/evaluation/eval_vqa_score.py --models stable-diffusion-3.5-large-turbo --modes concise,detailed --dialects sge bre
```

For CLIP Score evaluation:
```shell
python src/evaluation/eval_clip_score.py --models stable-diffusion-3.5-large-turbo,stable-diffusion3-medium --modes concise,detailed --dialects sge bre
```

##### Arguments

- `--models`: The list of models you want to evaluate.
- `--modes`: The list of modes you want to evaluate.
- `--dialects`: The dialects you want to evaluate, can be any of: [aae, bre, che, ine, sge].



#### Aggragate Evaluation Results

For aggregating evaluation results and calculating final scores for each dataset split, please refer to `src/evaluation/aggragate_model_scores.py` and `src/evaluation/calculate_split_scores.py`.


## 3. Mitigation





# BibTex
If you find our work helpful, please kindly cite our work :)
```bash
@article{zhou2025dialectgen,
  title={DialectGen: Benchmarking and Improving Dialect Robustness in Multimodal Generation},
  author={Zhou, Yu and An, Sohyun and Deng, Haikang and Yin, Da and Peng, Clark and Hsieh, Cho-Jui and Chang, Kai-Wei and Peng, Nanyun},
#   journal={arXiv preprint arXiv:},
  year={2025}
} 
```
