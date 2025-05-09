# Environmental Sound Classification: A Comparative Study of Deep Learning Architectures

This repository contains a comprehensive comparison of four different deep learning architectures for environmental sound classification using the ESC-50 dataset. The study evaluates the performance of both traditional and advanced neural network architectures to understand their effectiveness in sound classification tasks.

## Dataset and Preprocessing

### ESC-50 Dataset
The Environmental Sound Classification (ESC-50) dataset is a collection of 2,000 environmental audio recordings organized into 50 semantic classes. Each class contains 40 audio clips of 5 seconds each, sampled at 44.1 kHz. The dataset is organized into 5 major categories:
- Animals
- Natural soundscapes & water sounds
- Human, non-speech sounds
- Domestic sounds
- Exterior/urban noises

### Dataset Setup
1. Download the ESC-50 dataset from [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)

### Feature Extraction
All models use mel-spectrograms as their primary input features:

1. **Audio Preprocessing**
   - Resampling to 44.1 kHz (if needed)
   - Normalization to [-1, 1] range
   - Pre-emphasis filter (coefficient = 0.97)

2. **Spectrogram Generation**
   - Short-time Fourier transform (STFT) with:
     - Window size: 2048 samples (≈46.4ms)
     - Hop length: 512 samples (≈11.6ms)
     - Window function: Hann window
   - Mel-scale filterbank with 128 mel bands
   - Log-mel spectrograms computation
   - Input shape: (1, 128, 216) representing (channels, mel bands, time steps)

3. **Feature Storage**
   - Features are saved as numpy arrays
   - Organized by class and fold
   - Includes metadata for easy access

### Data Augmentation
The following augmentations are applied during training:
1. **Time Stretching**
   - Random stretching between 0.8x and 1.2x
   - Preserves pitch information

2. **Pitch Shifting**
   - Random shifting between -2 and +2 semitones
   - Preserves duration

3. **Background Noise**
   - Random noise addition (SNR between 10-20 dB)
   - Helps with robustness

4. **Time Shifting**
   - Random shifts up to 0.1 seconds
   - Helps with temporal invariance

## Models Under Comparison

### 1. ESC50CNN (Basic CNN)
A standard Convolutional Neural Network architecture designed as a baseline model.

**Architecture Details:**
```
Input (1, 128, 216)  # (channels, mel bands, time steps)
    │
    ▼
Conv2D (32 filters, 3x3) → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Conv2D (64 filters, 3x3) → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Conv2D (128 filters, 3x3) → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Global Average Pooling
    │
    ▼
Dense(256) → ReLU → Dropout(0.5) → Dense(50)
```

**Key Features:**
- Three convolutional blocks for spatial feature extraction
- Batch normalization for stable training
- Max pooling for dimensionality reduction
- Global average pooling for spatial invariance
- Dropout for regularization
- Focuses on capturing local patterns in spectrograms

### 2. ESC50CRNN (Convolutional Recurrent Neural Network)
Combines CNN layers with recurrent layers to capture both spatial and temporal features.

**Architecture Details:**
```
Input (1, 128, 216)
    │
    ▼
Conv2D (32 filters, 3x3) → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Conv2D (64 filters, 3x3) → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Conv2D (128 filters, 3x3) → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Reshape → Bidirectional GRU (256 units, 2 layers) → Dropout(0.5)
    │
    ▼
Dense(256) → ReLU → Dropout(0.5) → Dense(50)
```

**Key Features:**
- CNN layers extract spatial features
- Bidirectional GRU with 2 layers captures temporal dependencies
- Enhanced capability for sequential pattern recognition
- Better suited for sounds with temporal variations
- Temporal pooling for sequence length invariance

### 3. MSCNN (Multi-Scale CNN)
Advanced CNN architecture that processes input at multiple scales simultaneously.

**Architecture Details:**
```
Input (1, 128, 216)
    │
    ▼
MultiScale Block:
    ├─ Conv2D (32 filters, 3x3)
    ├─ Conv2D (32 filters, 5x5)
    ├─ Conv2D (32 filters, 7x7)
    └─ Concatenate → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
MultiScale Block:
    ├─ Conv2D (64 filters, 3x3)
    ├─ Conv2D (64 filters, 5x5)
    ├─ Conv2D (64 filters, 7x7)
    └─ Concatenate → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
MultiScale Block:
    ├─ Conv2D (128 filters, 3x3)
    ├─ Conv2D (128 filters, 5x5)
    ├─ Conv2D (128 filters, 7x7)
    └─ Concatenate → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Conv2D (128 filters, 1x1) → BatchNorm → ReLU
    │
    ▼
Global Average Pooling
    │
    ▼
Dense(256) → ReLU → Dropout(0.5) → Dense(50)
```

**Key Features:**
- Parallel convolutional paths with different kernel sizes (3x3, 5x5, 7x7)
- Multi-scale feature extraction at each block
- Channel concatenation for feature fusion
- 1x1 convolution for channel reduction
- Global average pooling for spatial invariance
- Improved ability to recognize patterns of varying durations

### 4. MSCRNN (Multi-Scale Convolutional Recurrent Neural Network)
Most sophisticated architecture combining multi-scale processing with recurrent layers.

**Architecture Details:**
```
Input (1, 128, 216)
    │
    ▼
MultiScale Block:
    ├─ Conv2D (32 filters, 3x3)
    ├─ Conv2D (32 filters, 5x5)
    ├─ Conv2D (32 filters, 7x7)
    └─ Concatenate → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
MultiScale Block:
    ├─ Conv2D (64 filters, 3x3)
    ├─ Conv2D (64 filters, 5x5)
    ├─ Conv2D (64 filters, 7x7)
    └─ Concatenate → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
MultiScale Block:
    ├─ Conv2D (128 filters, 3x3)
    ├─ Conv2D (128 filters, 5x5)
    ├─ Conv2D (128 filters, 7x7)
    └─ Concatenate → BatchNorm → ReLU → MaxPool2D (2x2)
    │
    ▼
Reshape → Bidirectional GRU (256 units, 2 layers) → Dropout(0.5)
    │
    ▼
Dense(256) → ReLU → Dropout(0.5) → Dense(50)
```

**Key Features:**
- Multi-scale feature extraction from CNN layers
- Parallel convolutional paths with different kernel sizes
- Temporal modeling through bidirectional GRU
- Channel concatenation for feature fusion
- Temporal pooling for sequence length invariance
- Most comprehensive feature extraction capability
- Best suited for complex environmental sounds

## Methodology

### Data Processing Pipeline

1. **Audio Preprocessing**
   - Resampling to 44.1 kHz (if needed)
   - Normalization to [-1, 1] range
   - Pre-emphasis filter (coefficient = 0.97)

2. **Feature Extraction**
   - Short-time Fourier transform (STFT) with:
     - Window size: 2048 samples (≈46.4ms)
     - Hop length: 512 samples (≈11.6ms)
     - Window function: Hann window
   - Mel-scale filterbank with 128 mel bands
   - Log-mel spectrograms computation
   - Input shape: (1, 128, 216) representing (channels, mel bands, time steps)

3. **Data Augmentation**
   - Time stretching (±20%)
   - Pitch shifting (±2 semitones)
   - Background noise addition (SNR: 10-20 dB)
   - Random time shifting (±0.1s)

### Training Protocol

1. **Cross-Validation Strategy**
   - 5-fold cross-validation using official ESC-50 folds
   - Training set: 1,600 samples per fold
   - Test set: 400 samples per fold
   - Validation split: 20% of training data

2. **Training Parameters**
   - Optimizer: Adam (learning rate = 0.001)
   - Learning rate scheduling: ReduceLROnPlateau
   - Batch size: 32
   - Maximum epochs: 100
   - Early stopping patience: 10 epochs
   - Loss function: Cross-entropy

3. **Regularization Techniques**
   - Dropout (0.5) in dense layers
   - Batch normalization after convolutions
   - Data augmentation during training
   - L2 regularization (weight decay = 1e-4)

### Evaluation Framework

1. **Primary Metrics**
   - Cross-validation accuracy
   - F1-score (macro and micro)
   - Precision and recall per class
   - Confusion matrices

2. **Performance Analysis**
   - Learning curves (training/validation)
   - ROC curves per class
   - Precision-recall curves
   - Error analysis by class
   - Feature visualization

3. **Statistical Validation**
   - Paired t-tests between models
   - Confidence intervals for accuracy
   - Statistical significance testing
   - Error distribution analysis

## Common Architecture Features

### Input Processing
- Sample rate: 44.1 kHz
- Mel spectrogram parameters:
  - N_MELS: 40 mel bands
  - N_FFT: 1024
  - HOP_LENGTH: 512 (50% overlap)
- Input shape: (1, 128, 216) representing (channels, mel bands, time steps)

### Training Parameters
- Optimizer: Adam with learning rate 0.001
- Learning rate scheduling: ReduceLROnPlateau
- Batch size: 32
- Number of epochs: 100
- Early stopping with patience of 10 epochs
- Cross-entropy loss function

## Training Methodology

### Cross-Validation
- 5-fold cross-validation using official ESC-50 folds
- Each fold contains 1,600 training samples and 400 test samples
- Validation split: 20% of training data

### Data Augmentation
- Time stretching (±20%)
- Pitch shifting (±2 semitones)
- Background noise addition
- Random time shifting

## Evaluation Metrics

### Primary Metrics
- Cross-validation accuracy (validation and test sets)
- Learning curves across epochs
- Loss metrics (training, validation, and test)

### Comprehensive Analysis
- Confusion matrices for class-wise performance
- ROC curves for each class
- Precision-recall curves
- F1-scores per class
- Class-wise error analysis

## Results and Analysis

The results of this comparison study can be found in the `reports/figures` directory, which includes:
- Individual model performance plots
- Comparative analysis across all models
- Learning curves and metrics visualization
- Comprehensive performance reports
- Error analysis and confusion matrices
- Class-wise performance breakdown

## Usage

To reproduce the results or run your own experiments:
1. Install the required dependencies
2. Download and preprocess the ESC-50 dataset
3. Run the training scripts for each model
4. Use the analysis scripts to generate comparison plots
5. Review the comprehensive reports in the reports directory

## Directory Structure

- `models/`: Contains the model architectures
- `training/`: Training scripts for each model
- `reports/`: Analysis results and visualizations
- `utils/`: Utility functions and helper scripts
- `scripts/`: Analysis and evaluation scripts
- `data/`: Dataset and preprocessed features
- `configs/`: Model and training configurations

## Requirements

- Python 3.x
- PyTorch
- torchaudio
- numpy
- matplotlib
- seaborn
- scikit-learn
- librosa
- soundfile
- tqdm

## Citation

If you use this code or results in your research, please cite:
[Add citation information here] 