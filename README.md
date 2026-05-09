# ♻️ Smart Waste Classification System using MobileNetV2
*An AI-powered computer vision system that automatically classifies waste images into recyclable categories to improve waste sorting efficiency.*

---

# 🏆 Project Tier
**Tier 2** – This project involves deep learning, transfer learning with MobileNetV2, image preprocessing, model evaluation, and real-world environmental applications. It goes beyond basic classification by implementing an optimized pretrained CNN architecture for practical waste management solutions.

---

# 🎯 Problem & Solution

## The Problem
Waste sorting is often performed manually, which is time-consuming and prone to human error. Many recyclable materials end up in landfills because people cannot correctly identify the appropriate waste category.

Incorrect waste disposal reduces recycling efficiency, increases environmental pollution, and creates additional costs for waste management facilities and municipalities.

## Our Solution
We developed an AI-powered computer vision system that automatically identifies waste categories from images using the MobileNetV2 deep learning model.

The system analyzes uploaded waste images and predicts whether the item belongs to categories such as cardboard, glass, metal, paper, plastic, or trash, helping users dispose of waste correctly.

## Impact
- Improves recycling accuracy and efficiency
- Reduces landfill waste
- Assists recycling centers and municipalities
- Saves manual sorting time and operational costs
- Promotes environmental sustainability and smart waste management

---

# 🔧 Technical Details

## Approach
- **Task**: Image Classification
- **Model**: MobileNetV2 (Transfer Learning)
- **Framework**: TensorFlow / Keras
- **Key Libraries**:
  - tensorflow
  - keras
  - numpy
  - matplotlib
  - scikit-learn
  - opencv-python

---

# 🧠 System Architecture

```text
Input Waste Image
        ↓
Image Preprocessing
(Resize, Normalize, Augmentation)
        ↓
MobileNetV2 Feature Extraction
        ↓
Dense Classification Layers
        ↓
Waste Category Prediction
        ↓
Display Result
