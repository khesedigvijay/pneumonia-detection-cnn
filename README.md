# Pneumonia Detection from Chest X-Rays 🫁

A deep learning project that classifies chest X-ray images as **Normal** or **Pneumonia**, built and trained end-to-end in Google Colab using TensorFlow/Keras.

## 🎯 Problem Statement
Pneumonia is a leading cause of death in children and the elderly worldwide, and early detection from chest X-rays can significantly improve treatment outcomes. This project explores whether a convolutional neural network can assist in flagging pneumonia cases from X-ray images.

> **Disclaimer:** This is an educational/portfolio project, not a certified medical diagnostic tool. Predictions should never be used for real clinical decisions.

## 📊 Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle, ~5,800 labeled X-ray images across Normal and Pneumonia classes.

## 🏗️ Approach
1. Preprocessed and augmented X-ray images (resize, normalize, rotation/zoom/flip augmentation)
2. Handled class imbalance using computed class weights
3. Trained a baseline CNN from scratch
4. Applied transfer learning with **MobileNetV2** (pretrained on ImageNet), then fine-tuned the top layers
5. Evaluated using accuracy, AUC-ROC, recall, confusion matrix, and classification report
6. Added **Grad-CAM** visualizations to explain which regions of the X-ray influenced each prediction

## 📈 Results

| Model | Accuracy | AUC | Recall |
|---|---|---|---|
| Baseline CNN | 87.2% | 0.91 | 0.85 |
| Fine-tuned MobileNetV2 | 93.8% | 0.97 | 0.96 |

*(Recall was prioritized over raw accuracy, since missing a true pneumonia case is the costlier error in a medical context.)*

**⚠️ Replace the table above with your own numbers** — run Step 18 from your notebook (`classification_report` output) and copy accuracy, AUC, and recall for both models.

## 🔍 Grad-CAM Example
![Grad-CAM Example](gradcam_example.png)

*The heatmap highlights the lung regions the model focused on when predicting pneumonia.*

## 🛠️ Tech Stack
- Python, TensorFlow / Keras
- Google Colab (GPU-accelerated training)
- OpenCV, scikit-learn, Matplotlib/Seaborn
- Gradio (for the live demo)

## 🚀 Try It Live
(https://ae9fe7d1d4304aac64.gradio.live)

## 📂 Repository Contents
- `pneumonia_detection.ipynb` — full training notebook (data prep → training → evaluation → Grad-CAM)
- `pneumonia_detector.h5` — trained model weights *(or link to Google Drive if too large)*
- `gradcam_example.png` — sample explainability output

## ▶️ How to Run
1. Open `pneumonia_detection.ipynb` in Google Colab
2. Add your own `kaggle.json` API key to download the dataset
3. Run all cells (GPU runtime recommended)

## 📌 Key Learnings
- Handling class imbalance is critical for medical datasets
- Transfer learning significantly outperforms training a CNN from scratch on small datasets
- Model explainability (Grad-CAM) is essential for building trust in medical AI applications

## 📬 Contact
Digvijay khese — https://www.linkedin.com/in/digvijay-khese-1b2b273

