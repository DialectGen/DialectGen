import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image # Import PIL for image loading if needed later
from torchvision import transforms
from torchvision.datasets.coco import CocoCaptions

# These are the standard mean/std values published for CLIP models (ViT-B/32)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
INPUT_RESOLUTION = 224 # Or the resolution your specific CLIP model expects

# # Canonical transform for evaluation/inference
# coco_transform = transforms.Compose([
#     transforms.Resize(INPUT_RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.CenterCrop(INPUT_RESOLUTION),
#     transforms.ToTensor(),
#     transforms.Normalize(CLIP_MEAN, CLIP_STD),
# ])


class MSCOCOCaptionDataset(Dataset):

    def __init__(self, caption_file_path, image_folder_path=None, transform=None):
        if not os.path.exists(caption_file_path):
             raise FileNotFoundError(f"Caption file not found: {caption_file_path}")

        with open(caption_file_path, 'r') as f:
            caption_data = json.load(f)

        self.image_folder_path = image_folder_path
        self.transform = transform # Store transform if provided

        self.captions_by_imageid = {}
        image_ids_with_captions = set() 

        print("Processing annotations...")
        num_processed = 0
        num_skipped = 0
        for ann in caption_data["annotations"]:
            img_id = ann.get("image_id")
            caption = ann.get("caption")

            # Skip if essential info is missing
            if img_id is None or caption is None:
                num_skipped += 1
                continue

            # Ensure caption is a non-empty string
            caption = str(caption).strip()
            if not caption:
                num_skipped += 1
                continue

            # Initialize list for new image_id
            if img_id not in self.captions_by_imageid:
                self.captions_by_imageid[img_id] = []

            # Add caption to the list for this image_id
            self.captions_by_imageid[img_id].append(caption)
            image_ids_with_captions.add(img_id)
            num_processed += 1

        # Store unique image IDs that have at least one valid caption, sorted
        self.image_ids = sorted(list(image_ids_with_captions))[:1000]

        print(f"Processed {num_processed} valid annotations, skipped {num_skipped}.")
        print(f"Found {len(self.image_ids)} unique images with captions.")


    def __len__(self):
        return len(self.image_ids)
    

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        captions = self.captions_by_imageid.get(image_id)

        item = {
            "image_id": image_id,
            "captions": captions
        }

        if self.image_folder_path:
            # Format image filename (12 digits, zero-padded)
            image_filename = f"{image_id:012d}.jpg"
            image_path = os.path.join(self.image_folder_path, image_filename)
            item["image_path"] = image_path # Store path even if loading fails

            image = Image.open(image_path)
            if self.transform:
               image = self.transform(image)
            item["image"] = image

        return item


def coco_collate_fn(batch):
    image_ids = []
    all_captions = []
    captions_per_image = []
    batch_indices = []
    images = [] # Uncomment if loading images

    if not batch: # Handle empty batch case
        return {'image_ids': [], 'all_captions': [], 'captions_per_image': [], 'batch_indices': []}

    for i, item in enumerate(batch):
        if not item or "image_id" not in item or "captions" not in item:
            print("Warning: Skipping invalid item in batch.")
            continue

        image_ids.append(item["image_id"])
        item_captions = item["captions"]
        captions_per_image.append(item_captions) # Keep grouped captions

        if item_captions: # Only extend if there are captions
            all_captions.extend(item_captions)
            # Add the batch index 'i' for each caption belonging to this image
            batch_indices.extend([i] * len(item_captions))

        if "image" in item and item["image"] is not None:
            images.append(item["image"])

    collated_batch = {
        'image_ids': image_ids,
        'all_captions': all_captions,
        'captions_per_image': captions_per_image, # Useful for per-image KL calculation
        'batch_indices': batch_indices # Maps flattened captions back to image index in batch
    }

    if images is not None:
       collated_batch['images'] = images
    return collated_batch