# CNN-Cats-vs-Dogs

Convolutional Neural Network (CNN) for binary image classification (cats vs dogs), built with TensorFlow/Keras. Trained on 10,000 images, achieving 83.1% test accuracy.

**Author:** Jakub Wojciechowski  
**Course:** Introduction to Python and Machine Learning, Kozminski University

---

## 1. Goal

Build and evaluate a CNN model for binary image classification. The model learns to distinguish between cat and dog images based on visual features extracted through convolutional layers.

## 2. Dataset

- **Source:** Cats vs Dogs (Kaggle)
- **Training set:** 10,000 images (5,000 cats, 5,000 dogs) — subset of 25,000 available
- **Image size:** 96×96 pixels (RGB)
- **Train/test split:** 90/10 with stratification

## 3. Model architecture

- 3 convolutional blocks: Conv2D → BatchNormalization → Conv2D → BatchNormalization → MaxPooling2D → Dropout
- Flatten → Dense(128, relu) → Dropout → Dense(1, sigmoid)
- Optimizer: Adam
- Loss: binary crossentropy
- Callbacks: EarlyStopping + ReduceLROnPlateau

## 4. Results

| Metric | Value |
|--------|-------|
| Test accuracy | **83.1%** |
| Test loss | 0.393 |
| Correct predictions | 831 / 1000 |
| Incorrect predictions | 169 / 1000 |

Training ran for 25 epochs (batch size 32, validation split 20%).

## 5. How to reproduce

```bash
pip install -r requirements.txt
```

1. Download the Cats vs Dogs dataset and place images in a `train/` folder
2. Open `JAKUB_WOJCIECHOWSKI_49872_PROJECT.ipynb` in Jupyter or VS Code
3. Run all cells in order

To use the pre-trained model without retraining:

```python
from keras.models import load_model
model = load_model('JAKUB_WOJCIECHOWSKI_49872_MODEL.keras')
```

To classify a single image:

```python
predict_one_image('My photos/cat.jpeg')
predict_one_image('My photos/dog.jpg')
```

## 6. Files

- `JAKUB_WOJCIECHOWSKI_49872_PROJECT.ipynb` — main notebook (data loading, model definition, training, evaluation)
- `JAKUB_WOJCIECHOWSKI_49872_MODEL.keras` — pre-trained model (83.1% test accuracy)
- `imports_for_ML.py` — helper functions including `load_images()` (course material)
- `My photos/` — sample images for single-image prediction (`cat.jpeg`, `dog.jpg`)
- `requirements.txt` — Python dependencies

Note: The `train/` folder with 10,000 training images is not included due to file size.
