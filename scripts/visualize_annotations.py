#!/usr/bin/env python3
"""
Visualize Ground Truth Annotations

This script reads images from images_download/ and their corresponding
Label Studio JSON annotations from annotated_images_download/, then creates
high-quality visualizations with bounding boxes and labels overlaid on the images.

Output: Saves annotated images to visualized_annotations/
"""

import os
import json
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

# Paths
IMAGES_DIR = "../images_download"
ANNOTATIONS_DIR = "../annotated_images_download"
OUTPUT_DIR = "../visualized_annotations"

# Color palette (using distinct, vibrant colors for different classes)
# Format: BGR for OpenCV
COLORS = [
    (52, 152, 219),   # Blue
    (231, 76, 60),    # Red
    (46, 204, 113),   # Green
    (155, 89, 182),   # Purple
    (241, 196, 15),   # Yellow
    (230, 126, 34),   # Orange
    (26, 188, 156),   # Turquoise
    (236, 240, 241),  # Light Gray
    (149, 165, 166),  # Dark Gray
    (192, 57, 43),    # Dark Red
]

# Visualization settings
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_PADDING = 5


def get_color_for_class(class_name, class_index):
    """Get a consistent color for each class."""
    return COLORS[class_index % len(COLORS)]


def visualize_annotations_pil(image_path, annotations, output_path):
    """Draw bounding boxes and labels using PIL for better quality."""
    # Read image using PIL for better quality
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return False

    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Try to use a nice font with smaller, more readable size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Collect all unique labels to assign colors
    all_labels = set()
    for ann in annotations:
        if ann.get("type") == "rectanglelabels":
            val = ann.get("value", {})
            labels = val.get("rectanglelabels", [])
            all_labels.update(labels)

    label_to_index = {label: i for i, label in enumerate(sorted(all_labels))}

    # Process each annotation
    for ann in annotations:
        if ann.get("type") != "rectanglelabels":
            continue

        val = ann.get("value", {})
        labels = val.get("rectanglelabels", [])

        # Convert Label Studio percentages to pixel coordinates
        x = int(val.get("x", 0) * w / 100.0)
        y = int(val.get("y", 0) * h / 100.0)
        box_w = int(val.get("width", 0) * w / 100.0)
        box_h = int(val.get("height", 0) * h / 100.0)

        # Skip invalid boxes (zero or negative dimensions)
        if box_w <= 0 or box_h <= 0:
            continue

        # Get color for this class
        if labels:
            label = labels[0]
            class_index = label_to_index.get(label, 0)
            color = get_color_for_class(label, class_index)
            # Convert BGR to RGB for PIL
            color_rgb = (color[2], color[1], color[0])
        else:
            color_rgb = (52, 152, 219)  # Default blue

        try:
            # Draw bounding box with thicker lines
            for thickness in range(BOX_THICKNESS):
                draw.rectangle(
                    [x + thickness, y + thickness, x + box_w - thickness, y + box_h - thickness],
                    outline=color_rgb,
                    width=1
                )

            # Draw label(s) with background
            for label in labels:
                # Get text bounding box
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Position text above the box (or below if too close to top)
                if y > text_height + TEXT_PADDING * 2:
                    text_x = x
                    text_y = y - text_height - TEXT_PADDING * 2
                else:
                    text_x = x
                    text_y = y + box_h + TEXT_PADDING

                # Draw semi-transparent background
                bg_rect = [
                    text_x,
                    text_y,
                    text_x + text_width + TEXT_PADDING * 2,
                    text_y + text_height + TEXT_PADDING * 2
                ]

                # Create a semi-transparent overlay
                overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(bg_rect, fill=(*color_rgb, 220))

                # Composite the overlay
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)

                # Draw text in white
                draw.text(
                    (text_x + TEXT_PADDING, text_y + TEXT_PADDING),
                    label,
                    fill=(255, 255, 255),
                    font=font
                )
        except Exception:
            # Skip annotations that cause drawing errors
            continue

    # Save with high quality (PNG for lossless, or high-quality JPEG)
    if output_path.lower().endswith('.png'):
        img.save(output_path, 'PNG', optimize=False)
    else:
        img.save(output_path, 'JPEG', quality=95, subsampling=0)

    return True


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Get all annotation files
    annotation_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*"))
    print(f"Found {len(annotation_files)} annotation files")

    processed = 0
    skipped = 0

    for ann_file in tqdm(annotation_files, desc="Processing images"):
        try:
            # Read annotation JSON
            with open(ann_file, 'r') as f:
                data = json.load(f)

            # Extract image info
            task = data.get("task", {})
            img_url = task.get("data", {}).get("image", "")
            img_name = os.path.basename(img_url)
            img_path = os.path.join(IMAGES_DIR, img_name)

            # Check if image exists
            if not os.path.exists(img_path):
                skipped += 1
                continue

            # Get annotations
            annotations = data.get("result", [])

            # Create output path
            output_path = os.path.join(OUTPUT_DIR, img_name)

            # Visualize and save
            if visualize_annotations_pil(img_path, annotations, output_path):
                processed += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            skipped += 1
            continue

    print(f"\n✅ Processing complete!")
    print(f"   Processed: {processed} images")
    print(f"   Skipped: {skipped} images")
    print(f"   Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
