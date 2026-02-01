# Face Mask Detection using ResNet50 Transfer Learning

A deep learning computer vision project that detects whether a person is wearing a face mask or not using ResNet50 pre-trained model with transfer learning techniques. This project achieves high accuracy by leveraging transfer learning from ImageNet-pre-trained weights.

##  Project Overview

This project implements a CNN-based computer vision solution to detect face masks in real-time. It uses **ResNet50**, a pre-trained deep convolutional neural network, and applies transfer learning to quickly train an accurate mask detector with limited computational resources.

The model is trained on a binary classification task:
- **Class 0**: Without Mask
- **Class 1**: With Mask
- ##  Features

- **Transfer Learning**: Leverages ResNet50 pre-trained on ImageNet
- **Binary Classification**: Accurately identifies masked vs unmasked faces
- **Image Preprocessing**: Automatic image resizing and normalization
- **Data Augmentation**: Rotation, zoom, shift, and flip augmentations for better generalization
- **Jupyter Notebook**: Complete end-to-end implementation
- **Easy to Deploy**: Simple prediction interface for new images
- **High Performance**: ~95%+ validation accuracy
- ##  Requirements

- Python 3.7+
- TensorFlow/Keras 2.x+
- NumPy
- Pillow (PIL)
- Scikit-learn
- Matplotlib
- Kaggle API
- ## 🚀 Installation

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/anjali-2310176/FACE-MASK-DETECTIONRESNET50-TRANSFER-LEARNING.git
cd FACE-MASK-DETECTIONRESNET50-TRANSFER-LEARNING
\`\`\`

### 2. Create a Virtual Environment (Optional but Recommended)
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Download Dataset
Set up Kaggle API credentials and download the dataset:
\`\`\`bash
kaggle datasets download -d omkargurav/face-mask-dataset --path . --unzip
\`\`\`
## 📁 Project Structure

\`\`\`
FACE-MASK-DETECTIONRESNET50-TRANSFER-LEARNING/
├── Face_Mask_Detection.ipynb       # Main Jupyter notebook
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── data/                          # Dataset folder (after download)
    ├── with_mask/
    └── without_mask/
\`\`\`

## 💻 Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
\`\`\`bash
jupyter notebook
\`\`\`

2. Open `Face_Mask_Detection.ipynb`

3. Run cells sequentially to:
   - Install dependencies
   - Download dataset
   - Load and preprocess images
   - Train the model
   - Evaluate performance
   - Make predictions

### Key Steps in the Notebook

#### Step 1: Install and Import Libraries
\`\`\`python
%pip install kaggle --upgrade --quiet
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
\`\`\`

#### Step 2: Download Dataset
\`\`\`python
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_key'
!kaggle datasets download -d omkargurav/face-mask-dataset --path . --unzip
\`\`\`

#### Step 3: Load and Preprocess Images
- Images are loaded from `with_mask/` and `without_mask/` folders
- Resized to 128x128 pixels
- Converted to RGB format
- Stored as NumPy arrays with corresponding labels

#### Step 4: Train-Test Split
\`\`\`python
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
\`\`\`

#### Step 5: Data Augmentation
\`\`\`python
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
\`\`\`

#### Step 6: Build Model with Transfer Learning
\`\`\`python
base_model = ResNet50(weights='imagenet', include_top=False, 
                      input_shape=(128, 128, 3))
# Freeze initial layers, unfreeze last convolutional block
# Add custom dense layers for binary classification
\`\`\`

#### Step 7: Train the Model
\`\`\`python
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=X_train.shape[0] // 32,
                    epochs=15,
                    validation_data=(X_test, y_test))
\`\`\`

#### Step 8: Make Predictions
\`\`\`python
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("With Mask")
else:
    print("Without Mask")
\`\`\`

##  Model Architecture

**Base Model**: ResNet50 (pre-trained on ImageNet)

**Custom Architecture**:
\`\`\`
Input (128 × 128 × 3)
    ↓
ResNet50 Base (Frozen initial layers + Unfrozen last block)
    ↓
Global Average Pooling
    ↓
Flatten
    ↓
Dense(128, activation='relu')
    ↓
Dropout (optional)
    ↓
Dense(1, activation='sigmoid') — Output
\`\`\`

**Model Configuration**:
- **Optimizer**: Adam (learning_rate=0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 15
- **Input Size**: 128 × 128 × 3

##  Training

**Transfer Learning Strategy**:
1. Load ResNet50 pre-trained on ImageNet
2. Freeze all initial layers to preserve learned features
3. Unfreeze the last convolutional block for fine-tuning
4. Add custom dense layers for binary mask classification
5. Train with a lower learning rate (0.0001)

**Data Augmentation**:
- Rotation (±20°)
- Zoom (±15%)
- Width/Height shift (±20%)
- Shear (±15%)
- Horizontal flip

This augmentation helps the model generalize better to unseen data.

##  Results

**Training Performance**:
- Training Accuracy: ~98%+
- Validation Accuracy: ~95%+
- Loss Convergence: Stable

**Visualizations**:
- Training vs Validation Accuracy plots
- Loss curves
- Confusion matrix (can be added)

##  How Transfer Learning Works

Transfer learning significantly improves training efficiency:

1. **Pre-trained Weights**: ResNet50 is already trained on 1.2 million ImageNet images to detect various features
2. **Feature Reuse**: Early layers learn generic features (edges, textures) applicable to any vision task
3. **Fine-tuning**: Later layers are trained on mask-specific features with limited data
4. **Faster Convergence**: Requires fewer epochs and less computational power

**Advantages**:
- Faster training
- Better accuracy with limited data
- Reduced computational cost
- Improved generalization

##  Future Improvements

- [ ] Real-time video processing with OpenCV
- [ ] Deploy as a web application (Flask/Django)
- [ ] Mobile app deployment (TensorFlow Lite)
- [ ] Multi-face detection in images
- [ ] Mask type classification (N95, surgical, cloth)
- [ ] Confidence score visualization
- [ ] GPU acceleration for inference
- [ ] Integrate with CCTV systems
- [ ] Add model explainability (Grad-CAM)
- [ ] REST API for predictions
  

## 👤 Author

**Anjali Singh**
- GitHub: [@anjali-2310176](https://github.com/anjali-2310176)
- Email: studentinstem1@gmail.com
- 

## 🙏 Acknowledgments

- ResNet50 architecture by Microsoft Research
- TensorFlow/Keras team for the excellent framework
- Kaggle for the Face Mask Dataset
- ImageNet dataset creators
- Open source community
