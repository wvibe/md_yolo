"""
Utility functions for data preparation and processing for YOLO models.
"""

import os
import glob
import random
from typing import List, Tuple, Dict, Optional, Union, Any

import cv2
import numpy as np
import torch
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from path and convert to RGB.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format
    """
    # Read image using OpenCV (in BGR format)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert from BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray, 
    target_size: Tuple[int, int], 
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize an image to the target size.
    
    Args:
        image: Input image as numpy array
        target_size: Target size as (width, height)
        keep_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        # Calculate the ratio
        h, w = image.shape[:2]
        target_w, target_h = target_size
        ratio = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Resize the image
        resized_img = cv2.resize(image, (new_w, new_h))
        
        # Create a black canvas of target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Compute offset to center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        
        return canvas
    else:
        # Just resize without preserving aspect ratio
        return cv2.resize(image, target_size)


def convert_yolo_bbox(
    bbox: Tuple[float, float, float, float], 
    img_width: int, 
    img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format bounding box to pixel coordinates.
    
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    Output: [x_min, y_min, x_max, y_max] (pixel coordinates)
    
    Args:
        bbox: Bounding box in YOLO format [x_center, y_center, width, height]
        img_width: Image width
        img_height: Image height
        
    Returns:
        Bounding box in [x_min, y_min, x_max, y_max] format
    """
    x_center, y_center, width, height = bbox
    
    # Convert to pixel values
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Calculate coordinates
    x_min = int(max(0, x_center - (width / 2)))
    y_min = int(max(0, y_center - (height / 2)))
    x_max = int(min(img_width, x_center + (width / 2)))
    y_max = int(min(img_height, y_center + (height / 2)))
    
    return x_min, y_min, x_max, y_max


def convert_to_yolo_format(
    bbox: Tuple[int, int, int, int], 
    img_width: int, 
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert [x_min, y_min, x_max, y_max] to YOLO format.
    
    Args:
        bbox: Bounding box in [x_min, y_min, x_max, y_max] format
        img_width: Image width
        img_height: Image height
        
    Returns:
        Bounding box in YOLO format [x_center, y_center, width, height]
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min
    
    # Calculate center coordinates
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height


def draw_bboxes(
    image: np.ndarray, 
    bboxes: List[Tuple[int, int, int, int]], 
    labels: List[int], 
    class_names: Optional[List[str]] = None,
    confidence_scores: Optional[List[float]] = None,
    color_mapping: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image
        bboxes: List of bounding boxes in [x_min, y_min, x_max, y_max] format
        labels: List of class labels (integers)
        class_names: Optional mapping from label IDs to class names
        confidence_scores: Optional list of confidence scores
        color_mapping: Optional mapping from label IDs to colors
        
    Returns:
        Image with bounding boxes drawn
    """
    img_copy = image.copy()
    
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x_min, y_min, x_max, y_max = bbox
        
        # Determine color
        if color_mapping and label in color_mapping:
            color = color_mapping[label]
        else:
            # Generate a random color if not provided
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Prepare label text
        if class_names and label < len(class_names):
            label_text = class_names[label]
        else:
            label_text = f"Class {label}"
            
        # Add confidence score if available
        if confidence_scores and i < len(confidence_scores):
            label_text += f" {confidence_scores[i]:.2f}"
            
        # Draw label background
        text_size = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )[0]
        cv2.rectangle(
            img_copy,
            (x_min, y_min - text_size[1] - 5),
            (x_min + text_size[0], y_min),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_copy,
            label_text,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
    return img_copy


def create_data_splits(
    data_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    extensions: List[str] = ['jpg', 'jpeg', 'png']
) -> Dict[str, List[str]]:
    """
    Create train/validation/test splits from a directory of images.
    
    Args:
        data_dir: Directory containing the images
        output_dir: Directory to save the split text files
        split_ratio: Ratio of train, validation, and test splits
        extensions: List of valid image extensions
        
    Returns:
        Dictionary with paths for each split
    """
    # Make sure ratios sum to 1
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image paths
    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(data_dir, f"*.{ext}")))
    
    # Shuffle images
    random.shuffle(all_images)
    
    # Calculate split sizes
    total = len(all_images)
    train_size = int(total * split_ratio[0])
    val_size = int(total * split_ratio[1])
    
    # Split images
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    # Create split files
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    # Write splits to files
    for split_name, image_list in splits.items():
        with open(os.path.join(output_dir, f"{split_name}.txt"), 'w') as f:
            for img_path in image_list:
                f.write(f"{img_path}\n")
    
    return splits


if __name__ == "__main__":
    # Simple test code
    print("Data utilities loaded successfully!") 