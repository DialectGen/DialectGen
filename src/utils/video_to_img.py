#!/usr/bin/env python3
"""
extract_video_frames.py
=======================

For every 0.mp4 stored at

    /local3/yuzhou/Dialect/multimodal-bias/data/video/{mode}/{dialect}/{model}/{*_videos}/{prompt}/0.mp4

this script extracts 9 frames (0 – 100 % of the timeline, inclusive) and saves them as
0.jpg … 8.jpg at the mirrored path under *image*:

    /local3/yuzhou/Dialect/multimodal-bias/data/image/{mode}/{dialect}/{model}/{*_imgs}/{prompt}/{0-8}.jpg
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# User-editable paths
VIDEO_ROOT  = Path("/local3/yuzhou/Dialect/multimodal-dialectal-bias/data/video")
IMAGE_ROOT  = Path("/local3/yuzhou/Dialect/multimodal-dialectal-bias/data/image")
VIDEO_NAME  = "0.mp4"
FRAME_COUNT = 9                         # 0 … 8
PERCENTS    = np.linspace(0.0, 1.0, FRAME_COUNT)  # 0 % → 100 %
# ---------------------------------------------------------------------


def find_videos(root: Path, filename: str = VIDEO_NAME):
    """Yield every Path to `filename` under `root`."""
    for mp4 in root.rglob(filename):
        if mp4.is_file():
            yield mp4


def build_image_path(video_path: Path) -> Path:
    """
    Convert
        …/data/video/<mode>/<dialect>/<model>/<xyz_videos>/<prompt>/0.mp4
    to
        …/data/image/<mode>/<dialect>/<model>/<xyz_imgs>/<prompt>/
    """
    parts = list(video_path.parts)
    try:
        idx = parts.index("video")
    except ValueError:
        raise RuntimeError(f"Unexpected path {video_path}")

    # rebuild path, swapping “video” → “image” and “*_videos” → “*_imgs”
    new_parts = parts[:]
    new_parts[idx] = "image"
    # folder just after model name (idx+4) is *_videos
    videos_folder = new_parts[idx + 4]
    new_parts[idx + 4] = videos_folder.replace("_videos", "_imgs")
    return Path(*new_parts[:-1])   # strip the trailing '0.mp4'


def extract_frames(mp4: Path, out_dir: Path):
    """Extract FRAME_COUNT frames from mp4 and write to out_dir/0.jpg … 8.jpg."""
    if (out_dir / "0.jpg").exists():          # assume all 9 already done
        return

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        print(f"⚠️  Could not open {mp4}", file=sys.stderr)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️  Empty video {mp4}", file=sys.stderr)
        cap.release()
        return

    targets = sorted({int(round(p * (total_frames - 1))) for p in PERCENTS})
    os.makedirs(out_dir, exist_ok=True)

    for i, f_idx in enumerate(targets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"⚠️  Failed to read frame {f_idx} in {mp4}", file=sys.stderr)
            continue
        cv2.imwrite(str(out_dir / f"{i}.jpg"), frame)

    cap.release()


def main():
    videos = list(find_videos(VIDEO_ROOT))
    for vp in tqdm(videos, desc="Processing videos"):
        out_path = build_image_path(vp)
        extract_frames(vp, out_path)


if __name__ == "__main__":
    main()
