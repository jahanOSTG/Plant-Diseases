# Plant Disease Classification using EfficientNetB0

## Project Overview
This project aims to classify **plant leaves** into three categories:
- Healthy
- Powdery Mildew
- Rust

We used **EfficientNetB0 (Transfer Learning)** with fine-tuning to build a high-accuracy classification model.

Dataset: [Kaggle: Plant Disease Recognition](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)

---

## Workflow

### 1. Data Preparation
- Dataset organized into `Train`, `Validation`, and `Test` folders.
- Classes: `Healthy`, `Powdery`, `Rust`
- Image augmentation applied to increase dataset diversity:
  - Rotation
  - Width/Height shift
  - Shear
  - Zoom
  - Horizontal/Vertical flip
- Preprocessing using **EfficientNetB0 preprocess_input**

### 2. Model Architecture
- **Base Model**: EfficientNetB0 (ImageNet pretrained, top removed)
- **Added Layers**:
  - GlobalAveragePooling2D
  - Dropout(0.4)
  - Dense(3, activation='softmax')
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

### 3. Training Strategy
- **Phase 1 (Feature Extraction)**: Freeze base model, train only added head layers
- **Phase 2 (Fine-tuning)**: Unfreeze top 50 layers of EfficientNetB0 for further training
- Callbacks used:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint

### 4. Results
- Validation Accuracy: 100% (best epoch)
- Test Accuracy: 93.3%
- Confusion Matrix & Classification Report show high precision and recall across all classes

### 5. Visualization
- Training/Validation Accuracy & Loss curves
- Confusion Matrix heatmap

### 6. Prediction on Custom Images
- Preprocess custom image (resize 224x224, preprocess_input)
- Use the trained model for prediction
- Display predicted class and probability scores

---

## Performance Metrics

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Healthy   | 0.93      | 0.86   | 0.90     | 50      |
| Powdery   | 0.90      | 0.94   | 0.92     | 50      |
| Rust      | 0.96      | 1.00   | 0.98     | 50      |

**Overall Test Accuracy: 93%**

---

## Key Takeaways
- Transfer learning with EfficientNetB0 gave excellent performance
- Data augmentation improved model generalization
- Fine-tuning boosted validation accuracy to 100%
- The trained model successfully predicts plant diseases from unseen images

---

## Tools Used

| Tool/Library | Purpose |
|--------------|---------|
| Python 3.x   | Programming language |
| TensorFlow & Keras | Deep learning framework |
| Google Colab | Cloud-based execution environment |
| NumPy        | Numerical operations |
| Pandas       | Data handling |
| Matplotlib   | Plotting & visualization |
| Seaborn      | Advanced plotting |
| PIL          | Image processing |

---

## Techniques Applied

| Technique | Description |
|-----------|-------------|
| Transfer Learning | Using EfficientNetB0 pretrained on ImageNet |
| Data Augmentation | Increase dataset diversity to prevent overfitting |
| Fine-tuning | Unfreeze top layers of base model for better accuracy |
| Early Stopping | Stop training when validation stops improving |
| Reduce LR on Plateau | Reduce learning rate when improvement stalls |
| Model Checkpoint | Save best model during training |
| Image Preprocessing | Resize, scale, and normalize input images |
| Prediction | Predict disease class for new images |

---

## References
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)
