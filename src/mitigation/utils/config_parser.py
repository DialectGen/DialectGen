from pathlib import Path

import datasets
import torch
import torch.optim as optim
import yaml
from datasets import load_dataset
from torch.nn.functional import cosine_similarity
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline


class ConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def load_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained(self._config['tokenizer'])
        return tokenizer

    def load_text_encoder(self):
        text_encoder = CLIPTextModel.from_pretrained(
            self._config['text_encoder'],
        )
        return text_encoder
    
    def load_encoder_and_tokenizer(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self._config["stable_diffusion_model"]
        )
        return pipe.text_encoder, pipe.tokenizer

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def create_loss_function(self):

        class SimilarityLoss(torch.nn.Module):

            def __init__(self, flatten: bool = False, reduction: str = 'mean'):
                super().__init__()
                self.flatten = flatten
                self.reduction = reduction

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                if self.flatten:
                    input = torch.flatten(input, start_dim=1)
                    target = torch.flatten(target, start_dim=1)

                loss = 1 - cosine_similarity(input, target, dim=1)

                if self.reduction == 'mean':
                    loss = loss.mean()
                elif self.reduction == 'sum':
                    loss = loss.sum()
                return loss

        loss_fkt = SimilarityLoss(flatten=True)
        return loss_fkt

    @property
    def clean_batch_size(self):
        return self.training['clean_batch_size']

    @property
    def polysemy_batch_size(self):
        return self.training['polysemy_batch_size']

    @property
    def stable_diffusion_model(self):
        return self._config['stable_diffusion_model']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def wandb(self):
        return self._config['wandb']
    
    @property
    def caption_dataset(self):
        return self._config['caption_dataset']
    
    @property
    def caption_file_path(self):
        return self.caption_dataset["caption_file_path"]
    
    @property
    def image_folder_path(self):
        return self.caption_dataset["image_folder_path"]
    
    @property
    def kl_batch_size(self):
        return int(self.caption_dataset["batch_size"])
    
    @property
    def kl_control_size(self):
        return int(self.caption_dataset["control_size"])
    
    @property
    def kl_control_size_eval(self):
        return int(self.caption_dataset["control_size_eval"])

    @property
    def kl_mode(self):
        return self.caption_dataset["mode"]
    
    @property
    def epochs(self):
        return self._config['training']['epochs']

    @property
    def hf_token(self):
        return self._config['hf_token']

    @property
    def loss_fkt(self):
        return self.create_loss_function()

    @property
    def dialect_file_folder(self):
        return self._config['dialect_file_folder']
    
    @property
    def clip_model(self):
        return self._config['clip_model']