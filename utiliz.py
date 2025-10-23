import tensorflow as tf
import numpy as np
from PIL import Image
import json

def process_image(image_path, image_size=224):
    """Load and preprocess an image for model prediction"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size))
    image_array = np.array(image) / 255.0  
    return np.expand_dims(image_array, axis=0)  

def load_category_names(json_path):
    """Load JSON file mapping class indices to flower names"""
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names
