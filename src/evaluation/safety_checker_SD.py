# Create file structure
import os
import sys
from transformers.models.poolformer.image_processing_poolformer import ImageInput
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Plot similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



# Load both text-to-image models
dtype = "float32"
import torch
from IPython.display import display, update_display
from min_dalle import MinDalle
from diffusers import StableDiffusionPipeline


import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel, CLIPImageProcessor

# ---------------------------------------- Basic Paths ----------------------------------------

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_ChicanoEnglish_v1"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/chicano_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_Singlish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/singlish_prompts.csv"


# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_BritishEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/british_english_prompts.csv"


# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_BritishEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/british_english_prompts.csv"

img_dir = "/local1/bryanzhou008/Dialect/data/expanded_aae_v1/"
data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/african_american_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/test_269_aae_entigen"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/aae269_entigen.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/test_269_aae"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/aae269.csv"

# ---------------------------------------------------------------------------------------------


# Global Variables
# library = "torchmetrics"
library = "openai"


# Read Data
df = pd.read_csv(data_file)
dialect_prompts = list(df["DialectPrompt"])
sae_prompts = list(df["SAEPrompt"])
















def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.register_buffer("concept_embeds_weights", torch.ones(17))
        self.register_buffer("special_care_embeds_weights", torch.ones(3))

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                images[idx] = np.zeros(images[idx].shape)  # black image

        # if any(has_nsfw_concepts):
        #     logger.warning(
        #         "Potential NSFW content was detected in one or more images. A black image will be returned instead."
        #         " Try again with a different prompt and/or seed."
        #     )

        return images, has_nsfw_concepts

    @torch.inference_mode()
    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # special_scores = special_scores.round(decimals=3)
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        images[has_nsfw_concepts] = 0.0  # black image

        return images, has_nsfw_concepts


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images



feature_extractor = CLIPImageProcessor()

def run_safety_checker(image, device, dtype):
    if safety_checker is not None:
        safety_checker_input = feature_extractor(numpy_to_pil(image), return_tensors="pt").to(device)
        image, nsfw_detected = safety_checker(
            images=image,
            clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
        )
    else:
        nsfw_detected = None
        watermark_detected = None

        # if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
        #     self.unet_offload_hook.offload()

    return image, nsfw_detected




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(device)





def get_nsfw_num(img_dir, model, dialect, gen_prompt, ref_prompt):
    prompt_dir = os.path.join(img_dir, model, dialect, gen_prompt)
    black_nsfw = 0
    for i in range(0, 9):
        img = np.array(Image.open(prompt_dir + f'/{i}.jpg'))
        img, nsfw_detected = run_safety_checker(img, device, dtype=torch.float32)
        # print("nsfw_detected:",nsfw_detected[0])
        if nsfw_detected[0] == True:
            black_nsfw += 1

    return black_nsfw










dialect_total_score_dalle = []
dialect_total_score_stable = []

sae_total_score_dalle = []
sae_total_score_stable = []

for i in tqdm(range(len(dialect_prompts))):
    # dialect
    dialect_prompt = dialect_prompts[i]
    print("dialectal prompt is:", dialect_prompt)

    # test
    sae_prompt = sae_prompts[i]
    print("sae prompt is:", sae_prompt)

    dialect_dalle_score = get_nsfw_num(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, sae_prompt)
    dialect_total_score_dalle.append(dialect_dalle_score)
    print("dialect_dalle_score:", dialect_dalle_score)

    dialect_stable_score = get_nsfw_num(img_dir, "stable-diffusion", "dialect_imgs", dialect_prompt, sae_prompt)
    dialect_total_score_stable.append(dialect_stable_score)
    print("dialect_stable_score:", dialect_stable_score)

    print("\n")

    # sae
    sae_prompt = sae_prompts[i]
    print("sae prompt is:", sae_prompt)


    sae_dalle_score = get_nsfw_num(img_dir, "minDALL-E", "sae_imgs", sae_prompt, sae_prompt)
    sae_total_score_dalle.append(sae_dalle_score)
    print("sae_dalle_score:", sae_dalle_score)


    sae_stable_score = get_nsfw_num(img_dir, "stable-diffusion", "sae_imgs", sae_prompt, sae_prompt)
    sae_total_score_stable.append(sae_stable_score)
    print("sae_stable_score:", sae_stable_score)


    print("\n")


print("-------------------final results-------------------")
print("dialect_avg_nsfw_dalle:", sum(dialect_total_score_dalle)/len(dialect_total_score_dalle)/9) 
print("dialect_avg_nsfw_stable:", sum(dialect_total_score_stable)/len(dialect_total_score_stable)/9)

print("sae_avg_nsfw_dalle:", sum(sae_total_score_dalle)/len(sae_total_score_dalle)/9)
print("sae_avg_nsfw_stable:", sum(sae_total_score_stable)/len(sae_total_score_stable)/9)

print("dialect_total_nsfw_dalle:", sum(dialect_total_score_dalle))
print("dialect_total_nsfw_stable:", sum(dialect_total_score_stable))
      
print("sae_total_nsfw_dalle:", sum(sae_total_score_dalle))
print("sae_total_nsfw_stable:", sum(sae_total_score_stable))