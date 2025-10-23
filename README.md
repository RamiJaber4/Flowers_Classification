# 🌸 Flower Image Classifier – TensorFlow & Transfer Learning

A deep learning project that classifies **102 different species of flowers** using **TensorFlow** and **TensorFlow Hub** with a **MobileNetV3** feature extractor.  
This project is part of the **Udacity "Intro to Machine Learning with TensorFlow" Nanodegree**.

---

## 📘 Overview

This project demonstrates how to:
- Load and preprocess image datasets using `TensorFlow Datasets`
- Build an **image classification pipeline**
- Apply **transfer learning** using a pre-trained **MobileNetV3** model
- Train a custom classifier head with **Batch Normalization**, **Dropout**, and **L2 regularization**
- Evaluate model performance and perform **inference** on unseen data

The trained model can predict the flower name from any input image.

---

## ⚙️ How It Works

1. **Dataset Loading** – Loads the [Oxford Flowers 102 dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from TensorFlow Datasets and splits it into training, validation, and test sets.  
2. **Preprocessing Pipeline** – Resizes each image to `224x224`, normalizes pixel values (0–1), and applies data augmentation (random rotation, flipping, brightness, and contrast).  
3. **Transfer Learning** – Uses a pre-trained `MobileNetV3-Large` model as a **feature extractor** (from TensorFlow Hub).  
4. **Custom Classifier** – Adds dense layers with ReLU activations, Batch Normalization, Dropout, and an output layer of 102 classes (softmax).  
5. **Training** – Trains only the new classifier layers using the Adam optimizer and monitors validation loss with EarlyStopping.  
6. **Evaluation** – Tests the trained model on unseen images and reports accuracy (~88% on test set).  
7. **Inference** – Loads saved model (`flower_classifier.h5`), processes any new image, and returns top-5 predicted flower names with probabilities.

---

## 🧠 Dataset

**Oxford Flowers 102 Dataset**  
Contains 102 flower categories with varying numbers of images per class.  
👉 [Dataset link](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

---

## 🧩 Model Architecture

- **Base Model:** MobileNetV3 Large (pre-trained on ImageNet)
- **Input Size:** 224 × 224 × 3
- **Classifier Head:**
  - Dense(1024) + BatchNorm + ReLU + Dropout(0.35)
  - Dense(512) + BatchNorm + ReLU + Dropout(0.45)
  - Dense(256) + BatchNorm + ReLU + Dropout(0.5)
  - Dense(102, activation='softmax')

- **Optimizer:** Adam (lr=0.00013)  
- **Callbacks:** EarlyStopping + ReduceLROnPlateau  
- **Best Validation Accuracy:** ~91%  
- **Test Accuracy:** ~88%

---

## 🧰 Requirements

Install dependencies:

```bash
tensorflow-datasets==4.9.4
tensorflow-hub==0.16.1
scipy==1.12.0
tf-keras==2.16.0 

```
## 🚀 How to Run

1️⃣ Train the model
python Project_Image_Classifier_Project.ipynb


or run it in Jupyter Notebook or Google Colab.

2️⃣ Make predictions

Use the command-line prediction script:

python predict.py --image_path test_images/orange_dahlia.jpg \
                  --model_path flower_classifier.h5 \
                  --category_names label_map.json \
                  --top_k 5


Example output:

Predictions:
orange dahlia : 0.94
wild pansy    : 0.03
pink primrose : 0.02

## 🖼 Example Output
Input Image	Top-5 Predictions

1️⃣ Rose
2️⃣ Tulip
3️⃣ Orchid
4️⃣ Daisy
5️⃣ Lily


## 👨‍💻 Author

Rami Jaber
📍 Computer Science Student – An-Najah National University
💼 GitHub: RamiJaber4


