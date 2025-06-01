import argparse
import os
from datetime import datetime
import torch
import torch.nn.functional as F
import wandb
from utils.config_parser import ConfigParser
import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd
import random
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.nn import KLDivLoss
from utils.hf_captions import create_hf_coco_dataset
from utils.misc import fix_seed
from const import *


def clip_inference(clip_model, clip_processor, caption_dataset_batch, mode="image"):
    """
    Compute similarity and embeddings for either image-text or text-text pairs using CLIP.

    Args:
        clip_model: a HuggingFace CLIPModel instance
        clip_processor: a HuggingFace CLIPProcessor instance
        caption_dataset_batch: dict with keys
            - "captions": a list of (text,) tuples
            - "image": a list of image tensors (only used in image mode)
        mode: "image" to compute logits_per_image & image_embeds (original behavior),
              "text"  to compute logits_per_text & text_embeds

    Returns:
        logits: similarity matrix (image-to-text or text-to-text)
        embeds: embeddings (image_embeds or text_embeds)
    """
    texts = [ct[0] for ct in caption_dataset_batch["captions"]]

    if mode == "image":
        images = caption_dataset_batch["image"]
        inputs = clip_processor(text=texts, images=images, return_tensors="pt", padding=True).to(clip_model.device)
        with torch.no_grad():
            outputs = clip_model(**inputs, return_dict=True)
        return outputs.logits_per_image, outputs.image_embeds

    elif mode == "text":
        # Tokenize only the texts
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(clip_model.device)
        with torch.no_grad():
            # Get text embeddings
            text_embeds = clip_model.get_text_features(**inputs)
        # Normalize embeddings to unit length
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # Compute pairwise text-to-text similarity matrix
        logit_scale = clip_model.logit_scale.exp().to(clip_model.device)
        logits_per_text = torch.matmul(text_embeds, text_embeds.t()) * logit_scale 
        
        return logits_per_text, text_embeds

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'image' or 'text'.")


def text_encoder_inference(tokenizer, text_encoder, caption_dataset_batch, device):
    """
    Compute similarity and embeddings for text-text pairs only. e.g., SD v2.1
    """
    texts = [ct[0] for ct in caption_dataset_batch["captions"]]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_embeds = text_encoder(**inputs)[1]
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logit_scale = 100.0
    logits_per_text = torch.matmul(text_embeds, text_embeds.t()) * logit_scale
    
    return logits_per_text, text_embeds


def main(args):
    config = ConfigParser(args.config)
    config_path = args.config
    fix_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training.get('num_threads', 4))

    # --- Load Dialect Data ---
    dataset = process_dialect_data(os.path.join(config.dialect_file_folder, args.mode), args.dialect)
    dataset = dataset.shuffle(seed=config.seed)
    
    # --- Load Polysemy Data ---
    polysemy_dataset = process_polysemy_data(os.path.join(config.dialect_file_folder, args.mode), args.dialect)
    polysemy_dataset = polysemy_dataset.shuffle(seed=config.seed)
    polysemy_train_loader = DataLoader(
        polysemy_dataset["train"],
        batch_size=config.polysemy_batch_size,
        shuffle=True,
        drop_last=False
    )
    polysemy_train_iter = iter(polysemy_train_loader)

    # --- Load Caption Dataset for Regularization ---
    kl_batch_size = config.kl_batch_size
    kl_control_size = config.kl_control_size
    kl_control_size_eval = config.kl_control_size_eval
    mscoco = create_hf_coco_dataset(CAPTION_FILE_PATH, IMAGE_FOLDER_PATH)
    caption_dataset = mscoco.select(range(kl_control_size))
    caption_dataset_eval = mscoco.select(range(kl_control_size, kl_control_size+kl_control_size_eval))

    # --- Load Models ---
    encoder_reference, tokenizer = config.load_encoder_and_tokenizer()
    encoder_reference = encoder_reference.to(device)
    encoder_policy, _ = config.load_encoder_and_tokenizer()
    encoder_policy = encoder_policy.to(device)

    # Freeze reference model
    for param in encoder_reference.parameters():
        param.requires_grad = False
    
    # --- Get logits & embeddings for KL ---
    if config.clip_model != "none":
        clip_processor = CLIPProcessor.from_pretrained(config.clip_model)
        clip_model = CLIPModel.from_pretrained(config.clip_model).to(device)
        for param in clip_model.parameters():
            param.requires_grad = False
        reference_logits, mode_embeds = clip_inference(clip_model, clip_processor, caption_dataset, mode=config.kl_mode)
        reference_logits_eval, mode_embeds_eval = clip_inference(clip_model, clip_processor, caption_dataset_eval, mode=config.kl_mode)
    else:
        reference_logits, mode_embeds = text_encoder_inference(tokenizer, encoder_reference, caption_dataset, device)
        reference_logits_eval, mode_embeds_eval = text_encoder_inference(tokenizer, encoder_reference, caption_dataset_eval, device)
    print(f"reference_logits shape: {reference_logits.shape}")
    print(f"mode_embeds shape: {mode_embeds.shape}")

    # --- Optimizer and Scheduler ---
    optimizer = config.create_optimizer(encoder_policy)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # --- Loss Functions ---
    loss_fkt = config.loss_fkt
    kl_loss_fn = KLDivLoss(reduction='batchmean', log_target=True)
    
    weight_unlearn = config.training.get('weight_unlearn', 1.0)
    weight_kl_reg = config.training.get('weight_kl_reg', 1.0)
    weight_polysemy_reg = config.training.get('weight_polysemy', 1.0)

    # --- WandB Logging ---
    if config.wandb['enable_logging']:
        lr = config.optimizer["AdamW"]["lr"]
        config.wandb['args']['name'] = f"e={config.training['epochs']}_lr={lr}_ul={weight_unlearn}_kl={weight_kl_reg}_size={kl_control_size}-{kl_batch_size}_mode={config.kl_mode}_ps={weight_polysemy_reg}-{config.polysemy_batch_size}"
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_policy)
        # Log more config details
        wandb.config.update({
            'optimizer_type': type(optimizer).__name__,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'training_epochs': config.training['epochs'],
            'seed': config.seed,
            'dialect': args.dialect,
            'weight_kl_reg': weight_kl_reg,
            'weight_polysemy_reg': weight_polysemy_reg,
            'kl_batch_size': kl_batch_size,
            'kl_control_size': kl_control_size
        })

    # --- Set save config ---
    save_path_base = config.training.get('save_path', 'models')
    run_id = wandb_run.id if config.wandb['enable_logging'] and 'wandb_run' in locals() else f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name = f"{run_id}_{config.wandb['args']['name']}"
    model_name = config.stable_diffusion_model.split("/")[-1]
    save_dir = os.path.join(save_path_base, "finetune", model_name, args.mode, args.dialect)

    # --- Training Preparation ---
    step = 0
    encoder_policy.train()
    encoder_reference.eval()
    best_eval_loss = float('inf') 
    
    # --- Training Loop ---
    print(f'Starting Training for {config.training["epochs"]} epochs...')
    for ep in range(config.training['epochs']):
        print(f"\n--- Epoch {ep+1}/{config.training['epochs']} ---")
        encoder_policy.train() # Set model to train mode each epoch
        total_epoch_loss = 0.0
        total_epoch_unlearn_loss = 0.0
        total_epoch_kl_loss = 0.0
        total_epoch_polysemy_reg_loss = 0.0

        num_batches = int(np.ceil(len(dataset["train"])/config.clean_batch_size))

        for i in range(num_batches):
            batch = dataset["train"][i*config.clean_batch_size:(i+1)*config.clean_batch_size]
            batch_sae_prompt = batch['sae_prompts']
            batch_dialect_prompt = batch['dialect_prompts']
            
            try:
                batch_polysemy = next(polysemy_train_iter)
            except StopIteration:
                polysemy_train_iter = iter(polysemy_train_loader)
                batch_polysemy = next(polysemy_train_iter)
            batch_polysemy_prompt = batch_polysemy["polysemy_prompts"]

            sae_input = tokenizer(
                batch_sae_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)
            dialect_input = tokenizer(
                batch_dialect_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)
            polysemy_input = tokenizer(
                batch_polysemy_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)

            # --- Forward Pass ---
            embed_policy_dialect = encoder_policy(dialect_input.input_ids)[0]
            embed_policy_polysemy = encoder_policy(polysemy_input.input_ids)[0]
            
            with torch.no_grad():
                embed_reference_sae = encoder_reference(sae_input.input_ids)[0]
                embed_reference_polysemy = encoder_reference(polysemy_input.input_ids)[0]

            loss_unlearn = loss_fkt(embed_reference_sae, embed_policy_dialect)
            loss_polysemy_reg = loss_fkt(embed_policy_polysemy, embed_reference_polysemy)

            # --- KL Divergence Loss Calculation ---
            text_embeds = []
            for j in range(int(np.ceil(kl_control_size/kl_batch_size))):
                batch = caption_dataset[j*kl_batch_size:(j+1)*kl_batch_size]
                ts = [ct[0] for ct in batch["captions"]]
                inputs = tokenizer(
                    ts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                text_output = encoder_policy(**inputs)[1]
                text_embeds.append(text_output)
            
            text_embeds = torch.cat(text_embeds, dim=0)
            
            if config.clip_model != "none":
                text_embeds = clip_model.text_projection(text_embeds) # [1000, 768]
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) # [1000, 768]
                print(f'text_embeds shape: {text_embeds.shape}')

                # cosine similarity as logits
                logit_scale = clip_model.logit_scale.exp() # 100.0
                logits_per_text = torch.matmul(text_embeds, mode_embeds.t()) * logit_scale # [1000, 1000]
                logits_per_image = logits_per_text.T
            else:
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) # [1000, 768]
                print(f'text_embeds shape: {text_embeds.shape}')

                # cosine similarity as logits
                logit_scale = 100.
                logits_per_text = torch.matmul(text_embeds, mode_embeds.t()) * logit_scale # [1000, 1000]
                logits_per_image = logits_per_text.T
            
            kl_dim = 0     # for column-wise kl (per caption)
            loss_kl_reg = kl_loss_fn(
                F.log_softmax(logits_per_image, dim=kl_dim),
                F.log_softmax(reference_logits, dim=kl_dim)
            )
            
            if logits_per_image.isnan().any():
                print("NaN detected in logits_per_image")
                exit()

            loss = weight_unlearn * loss_unlearn + weight_kl_reg * loss_kl_reg + weight_polysemy_reg * loss_polysemy_reg

            # --- Backpropagation and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Logging ---
            loss_val = loss.item()
            loss_unlearn_val = loss_unlearn.item()
            loss_kl_reg_val = loss_kl_reg.item()
            loss_polysemy_reg_val = loss_polysemy_reg.item()

            total_epoch_loss += loss_val
            total_epoch_unlearn_loss += loss_unlearn_val
            total_epoch_kl_loss += loss_kl_reg_val
            total_epoch_polysemy_reg_loss += loss_polysemy_reg_val

            print(
                f'Epoch {ep+1}/{config.training["epochs"]} | Step {step} | Batch {i+1}/{num_batches} | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
                f'Loss: {loss_val:.4f} | Unlearn: {loss_unlearn_val:.4f} | '
                f'KL Reg: {loss_kl_reg_val:.4f} | Polysemy Reg: {loss_polysemy_reg_val:.4f}'
            )
            if config.wandb['enable_logging']:
                wandb.log({
                    'step': step,
                    'epoch': ep + (i / num_batches), # Fractional epoch
                    'train_loss': loss_val,
                    'train_loss_unlearn': loss_unlearn_val,
                    'train_loss_kl_reg': loss_kl_reg_val,
                    'train_loss_reg_polysemy': loss_polysemy_reg_val,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'weight_unlearn': weight_unlearn,
                    'weight_kl_reg': weight_kl_reg,
                    'weight_polysemy_reg': weight_polysemy_reg,
                })
            step += 1 # Increment global step counter
        
        # --- Learning Rate Scheduling ---
        if lr_scheduler:
            lr_scheduler.step() # Check if scheduler steps per batch or per epoch

        # --- End of Epoch ---
        avg_epoch_loss = total_epoch_loss / num_batches
        print(f"--- Epoch {ep+1} Summary ---")
        print(f"Average Train Loss: {avg_epoch_loss:.4f}")

        # --- VALIDATION ---
        print("Running Validation...")
        encoder_policy.eval() # Set model to evaluation mode
        total_eval_loss = 0.0
        total_eval_unlearn_loss = 0.0
        total_eval_kl_loss = 0.0
        total_eval_polysemy_reg_loss = 0.0
        # Use the validation split of the dialect dataset
        eval_batch_data = dataset["validation"][:] # Get all validation data (adjust if too large)
        eval_batch_polysemy_prompt = polysemy_dataset["validation"]["polysemy_prompts"]

        sae_input_eval = tokenizer(
            eval_batch_data['sae_prompts'], padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)
        dialect_input_eval = tokenizer(
            eval_batch_data['dialect_prompts'], padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)
        polysemy_input_eval = tokenizer(
            eval_batch_polysemy_prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)

        with torch.inference_mode(): # Use inference mode for validation
            # Policy model outputs
            embed_policy_dialect_eval = encoder_policy(dialect_input_eval.input_ids)[0]
            embed_policy_polysemy_eval = encoder_policy(polysemy_input_eval.input_ids)[0]

            # Reference model outputs
            embed_reference_sae_eval = encoder_reference(sae_input_eval.input_ids)[0]
            embed_reference_dialect_eval = encoder_reference(dialect_input_eval.input_ids)[0]
            embed_reference_polysemy_eval = encoder_reference(polysemy_input_eval.input_ids)[0]

        # Calculate validation losses
        loss_unlearn_eval = loss_fkt(embed_reference_sae_eval, embed_policy_dialect_eval)
        loss_polysemy_reg_eval = loss_fkt(embed_policy_polysemy_eval, embed_reference_polysemy_eval)

        # KL Divergence Loss (using train kl loss for simplicity)
        text_embeds_eval = []
        for j in range(int(np.ceil(kl_control_size_eval/kl_batch_size))):
            batch = caption_dataset_eval[j*kl_batch_size:(j+1)*kl_batch_size]
            ts = [ct[0] for ct in batch["captions"]]
            inputs_eval = tokenizer(
                ts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with torch.inference_mode():
                text_output_eval = encoder_policy(**inputs_eval)[1]
            text_embeds_eval.append(text_output_eval)
        
        text_embeds_eval = torch.cat(text_embeds_eval, dim=0)
        
        if config.clip_model != "none":
            text_embeds_eval = clip_model.text_projection(text_embeds_eval) # [1000, 768]
            text_embeds_eval = text_embeds_eval / text_embeds_eval.norm(p=2, dim=-1, keepdim=True) # [1000, 768]
            print(f'text_embeds_eval shape: {text_embeds_eval.shape}')
            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp() # 100.0
            logits_per_text_eval = torch.matmul(text_embeds_eval, mode_embeds_eval.t()) * logit_scale # [1000, 1000]
            logits_per_image_eval = logits_per_text_eval.T
        else:
            text_embeds_eval = text_embeds_eval / text_embeds_eval.norm(p=2, dim=-1, keepdim=True) # [1000, 768]
            print(f'text_embeds_eval shape: {text_embeds_eval.shape}')
            # cosine similarity as logits
            logit_scale = 100.
            logits_per_text_eval = torch.matmul(text_embeds_eval, mode_embeds_eval.t()) * logit_scale # [1000, 1000]
            logits_per_image_eval = logits_per_text_eval.T

        loss_kl_reg_eval = kl_loss_fn(
            F.log_softmax(logits_per_image_eval, dim=kl_dim),
            F.log_softmax(reference_logits_eval, dim=kl_dim)
        )

        # Total validation loss
        eval_loss = weight_unlearn * loss_unlearn_eval + weight_kl_reg * loss_kl_reg_eval + weight_polysemy_reg * loss_polysemy_reg_eval

        total_eval_loss = eval_loss.item()
        total_eval_unlearn_loss = loss_unlearn_eval.item()
        total_eval_kl_loss = loss_kl_reg_eval.item()
        total_eval_polysemy_reg_loss = loss_polysemy_reg_eval.item()

        print(
            f'Validation Loss: {total_eval_loss:.4f} | Unlearn: {total_eval_unlearn_loss:.4f} | '
            f'KL Reg: {total_eval_kl_loss:.4f} | Polysemy Reg: {total_eval_polysemy_reg_loss:.4f}'
        )
        if config.wandb['enable_logging']:
            wandb.log({
                'step': step,
                'epoch': ep + 1,
                'eval_loss': total_eval_loss,
                'eval_loss_unlearn': total_eval_unlearn_loss,
                'eval_loss_kl_reg': total_eval_kl_loss,
                'eval_loss_polysemy_reg': total_eval_polysemy_reg_loss,
            })

        if total_eval_loss < best_eval_loss:
            best_eval_loss = total_eval_loss
            best_epoch = ep + 1
            best_save_path = os.path.join(save_dir, name, "best")
            os.makedirs(best_save_path, exist_ok=True)
            encoder_policy.save_pretrained(best_save_path)
            tokenizer.save_pretrained(best_save_path)
            print(f"--> New best model saved (Epoch {best_epoch}, Loss {best_eval_loss:.4f}) to: {best_save_path}")

    # --- Save Trained Model ---
    save_path = os.path.join(save_dir, name, "last")

    os.makedirs(save_path, exist_ok=True)
    encoder_policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")

    if config.wandb['enable_logging'] and 'wandb_run' in locals():
        model_artifact = wandb.Artifact(f'policy_encoder_{args.dialect}', type='model')
        model_artifact.add_dir(save_path)
        wandb_run.log_artifact(model_artifact)
        wandb.summary['model_save_path'] = save_path
        wandb.summary['final_eval_loss'] = total_eval_loss
        wandb.finish()


# --- Data Processing Functions (Keep as they are) ---
def process_dialect_data(folder, dialect_str):
    dialect_list = dialect_str.split(",")
    
    split_dict = {"train": [], "val": [], "test": []}
    for dialect in dialect_list:
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(folder, dialect, f'{split}.csv')
            try:
                df = pd.read_csv(file_path, encoding="unicode_escape")
                data_dict = {
                    "dialect_words": df["Dialect_Word"].astype(str).tolist(),
                    "sae_words": df["SAE_Word"].astype(str).tolist(),
                    "dialect_prompts": df["Dialect_Prompt"].astype(str).tolist(),
                    "sae_prompts": df["SAE_Prompt"].astype(str).tolist()
                }
                # split_dict[split] = Dataset.from_dict(data_dict)
                split_dict[split].append(data_dict)
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}. Skipping split '{split}'.")
            except KeyError as e:
                print(f"Error: Column {e} not found in {file_path}. Please check CSV headers.")
                raise
    
    merged_split_dict = {}
    for split in ["train", "val", "test"]:
        if split_dict[split]:
            merged_data = {key: sum([d[key] for d in split_dict[split]], []) for key in split_dict[split][0].keys()}
            merged_split_dict[split] = Dataset.from_dict(merged_data)
        else:
            merged_split_dict[split] = Dataset.from_dict({
                "dialect_words": [], "sae_words": [], "dialect_prompts": [], "sae_prompts": []
            })
    return DatasetDict({
        "train": merged_split_dict["train"],
        "validation": merged_split_dict["val"],
        "test": merged_split_dict["test"]
    })


def process_polysemy_data(folder, dialect_str):
    dialect_list = dialect_str.split(",")
    
    split_dict = {'train': [], 'val': [], 'test': []}
    for dialect in dialect_list:
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(folder, dialect, f'{split}.csv')
            try:
                df = pd.read_csv(file_path, encoding="unicode_escape")
                polysemic = df["polysemic"].tolist()
                polysemy_prompts = [item for i, item in enumerate(df["Polysemy_Prompt"].tolist()) if polysemic[i]]
                split_dict[split].append({"polysemy_prompts": polysemy_prompts})
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}. Skipping split '{split}' for dialect '{dialect}'.")
            except KeyError as e:
                print(f"Error: Column {e} not found in {file_path}. Please check CSV headers.")
                raise

    merged_split_dict = {}
    for split in ['train', 'val', 'test']:
        if split_dict[split]:
            merged_prompts = sum([d["polysemy_prompts"] for d in split_dict[split]], [])
            merged_split_dict[split] = Dataset.from_dict({"polysemy_prompts": merged_prompts})
        else:
            merged_split_dict[split] = Dataset.from_dict({"polysemy_prompts": []})

    return DatasetDict({
        "train": merged_split_dict["train"],
        "validation": merged_split_dict["val"],
        "test": merged_split_dict["test"]
    })


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dialect Unlearning with KL Regularization')
    parser.add_argument('-c',
                        '--config',
                        default="configs/sd15.yaml",
                        type=str,
                        dest="config",
                        help='Config .yaml file path (default: configs/sd15.yaml)')
    parser.add_argument('--dialect', type=str, default='aae,bre,che,ine,sge', help="For multiple dialects, use comma. e.g., aae,bre")
    parser.add_argument('--mode', type=str, default="concise")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)