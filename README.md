# 🧠 Automated Psychological Analysis of Children's Drawings Using Machine Learning

![POSTER](https://github.com/user-attachments/assets/85100f81-0418-487b-bbeb-60b9a17cea03)


## 🎯 Project Overview

This project utilizes **machine learning and deep learning** techniques to analyze children's drawings and **automatically infer their psychological states**—including emotions like happiness, sadness, fear, and anger.

Traditionally, psychologists interpret children's drawings to gain insights into their emotional well-being. Our project introduces **automation** into this domain, making the process more **efficient, objective, and scalable**.

> 🧒 “Children may not always explain what they feel. But their drawings often do.”

---

## 📌 Key Objectives

- 📂 Categorize children's drawings into **four emotional states**: Happy, Sad, Angry, Fearful.
- 🧪 Compare traditional ML models vs. CNNs in emotion classification.
- ⚙️ Design a flexible framework to assist **psychologists and caregivers** in emotion assessment.

---

## 🖼️ Dataset

- Source: [Kaggle - Children’s Drawings Dataset](https://www.kaggle.com/)  
- ~700 images divided into:
  - **Happy**: Bright colors, smiling faces, harmonious scenes.
  - **Sad**: Cool tones, rain, empty spaces.
  - **Angry**: Bold red/black strokes, chaotic shapes.
  - **Fear**: Abstract, dark, distorted compositions.
- Preprocessed to **128x128 resolution**.

---

## 🧪 Methodology

### 🎨 Feature Engineering (Traditional ML Models)
- **Color Analysis** using HSV histograms.
- **Shape Analysis** using contour detection and area statistics.
- Extracted features fed into classifiers: Logistic Regression, SVM, Random Forest, Gradient Boosting, etc.

### 🤖 Deep Learning (CNN)
- Raw images processed with **data augmentation** and **ImageDataGenerator**.
- Custom CNN built with:
  - 3 Convolutional Layers
  - MaxPooling, Dense, Dropout
  - Trained using **Adam optimizer** & **categorical crossentropy**
- Visualized training progress with accuracy curves.

---

## 📊 Results Summary

| Model                   | Accuracy |
|-------------------------|----------|
| **Gradient Boosting**   | 0.59     |
| **Random Forest**       | 0.57     |
| Logistic Regression     | 0.50     |
| CNN (Validation)        | 0.43     |
| Naive Bayes             | 0.35     |

> 🎯 **CNN outperformed all in feature learning capacity**, though traditional models showed better raw accuracy due to handcrafted features.

---

## 📈 Visuals & Plots

### CNN Training vs Validation Accuracy

![WhatsApp Image 2025-06-26 at 13 07 18](https://github.com/user-attachments/assets/f8b9ddf5-56ff-4835-9f5d-92b6a6d0f556)
