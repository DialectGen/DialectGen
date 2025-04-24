#!/usr/bin/env python3
import os
import glob
import shutil
import tarfile
import tempfile

# Configuration paths
TEXT_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/concise"
IMAGE_BASE_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/concise"
CACHE_DIR = "/local1/bryanzhou008/cache"

# Create a temporary directory for collecting all SD2.1 images
temp_dir = tempfile.mkdtemp(dir=CACHE_DIR, prefix="sd21_images_")
print(f"Copying all Stable Diffusion 2.1 images into: {temp_dir}")

# Iterate over each dialect folder (based on CSV filenames)
for csv_file in os.listdir(TEXT_DIR):
    if not csv_file.lower().endswith('.csv'):
        continue
    dialect_name = os.path.splitext(csv_file)[0]
    sd21_dir = os.path.join(IMAGE_BASE_DIR, dialect_name, "stable-diffusion2.1")

    if not os.path.isdir(sd21_dir):
        print(f"Warning: SD2.1 directory for '{dialect_name}' not found. Skipping.")
        continue

    # Copy images from both dialect_imgs and sae_imgs
    for subfolder in ["dialect_imgs", "sae_imgs"]:
        sub_dir = os.path.join(sd21_dir, subfolder)
        if not os.path.isdir(sub_dir):
            print(f"Warning: '{subfolder}' not found under {sd21_dir}. Skipping.")
            continue

        # For each prompt folder within this subfolder
        for prompt_name in os.listdir(sub_dir):
            prompt_dir = os.path.join(sub_dir, prompt_name)
            if not os.path.isdir(prompt_dir):
                continue

            # Copy all .jpg images (0.jpg to 8.jpg)
            jpg_files = glob.glob(os.path.join(prompt_dir, "*.jpg"))
            if not jpg_files:
                print(f"Warning: No JPG images in {prompt_dir}")
                continue
            for src_path in jpg_files:
                try:
                    shutil.copy(src_path, temp_dir)
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")

# Create a tar.gz archive of the collected images
tar_name = os.path.basename(temp_dir) + ".tar.gz"
archive_path = os.path.join(CACHE_DIR, tar_name)
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(temp_dir, arcname=os.path.basename(temp_dir))

print(f"Archive created at: {archive_path}")
