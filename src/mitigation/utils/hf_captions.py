import datasets
from datasets import Features, Value, Sequence, Image as HFImage # Renamed PIL.Image to avoid conflict
import json
import os

def generate_coco_examples(caption_file_path: str, image_folder_path: str):
    """
    Generator function to yield examples for the Hugging Face Dataset.
    Performs the preprocessing similar to the original __init__ and yields
    data similar to __getitem__.
    """
    if not os.path.exists(caption_file_path):
        raise FileNotFoundError(f"Caption file not found: {caption_file_path}")
    if not os.path.isdir(image_folder_path):
        raise FileNotFoundError(f"Image folder not found: {image_folder_path}")

    with open(caption_file_path, 'r') as f:
        caption_data = json.load(f)

    captions_by_imageid = {}
    image_ids_with_captions = set()

    print("Processing annotations for generator...")
    num_processed = 0
    num_skipped = 0
    for ann in caption_data["annotations"]:
        img_id = ann.get("image_id")
        caption = ann.get("caption")

        if img_id is None or caption is None:
            num_skipped += 1
            continue

        caption = str(caption).strip()
        if not caption:
            num_skipped += 1
            continue

        if img_id not in captions_by_imageid:
            captions_by_imageid[img_id] = []

        captions_by_imageid[img_id].append(caption)
        image_ids_with_captions.add(img_id)
        num_processed += 1

    # Store unique image IDs that have at least one valid caption, sorted
    image_ids = sorted(list(image_ids_with_captions))

    print(f"Generator: Processed {num_processed} valid annotations, skipped {num_skipped}.")
    print(f"Generator: Found {len(image_ids)} unique images with captions.")

    # Yield data for each image_id
    for image_id in image_ids:
        captions = captions_by_imageid.get(image_id, []) # Get captions, default to empty list
        if not captions: # Should not happen with the filtering above, but safe check
             continue

        # Format image filename (12 digits, zero-padded)
        image_filename = f"{image_id:012d}.jpg"
        image_path = os.path.join(image_folder_path, image_filename)

        # Check if image file actually exists before yielding
        if os.path.exists(image_path):
            # The HF Image feature will load the image from this path
            yield {
                "image_id": image_id,
                "captions": captions,
                "image": image_path  # Provide the path for the Image feature
            }
        else:
            print(f"Warning: Image file not found, skipping: {image_path}")


def create_hf_coco_dataset(caption_file_path, image_folder_path):
    """
    Creates the Hugging Face Dataset using the generator.
    """
    # Define the structure of the dataset
    # The 'image' feature type automatically handles loading PIL Images from paths
    features = Features({
        'image_id': Value('int64'),
        'captions': Sequence(Value('string')),
        'image': HFImage() # Use datasets.Image feature
    })

    # Create the dataset using the generator
    # gen_kwargs passes arguments to the generator function
    hf_dataset = datasets.Dataset.from_generator(
        generate_coco_examples,
        features=features,
        gen_kwargs={
            "caption_file_path": caption_file_path,
            "image_folder_path": image_folder_path,
        },
        cache_dir="./data",
    )
    return hf_dataset