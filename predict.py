import argparse
import tensorflow as tf
import numpy as np
from utils import process_image, load_category_names

def predict(image_path, model_path, top_k=5, category_names=None):
    
    model = tf.keras.models.load_model(model_path, compile=False)

   
    image = process_image(image_path)

   
    preds = model.predict(image)[0] 
    top_indices = preds.argsort()[-top_k:][::-1]  
    top_probs = preds[top_indices]

    
    if category_names:
        top_classes = [category_names[str(i)] for i in top_indices]
    else:
        top_classes = top_indices.tolist()

    return top_probs, top_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower class from an image using a trained Keras model.')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model (.h5)')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file mapping labels to flower names')

    args = parser.parse_args()

    category_names = load_category_names(args.category_names) if args.category_names else None

    probs, classes = predict(args.image_path, args.model_path, args.top_k, category_names)

    print('Predictions:')
    for cls, prob in zip(classes, probs):
        print(f"{cls}: {prob:.4f}")
