# Animal Classifier CNN

A PyTorch-based Convolutional Neural Network (CNN) for classifying animal images into 10 categories:
- cane (dog)
- cavallo (horse)
- elefante (elephant)
- farfalla (butterfly)
- gallina (chicken)
- gatto (cat)
- mucca (cow)
- pecora (sheep)
- ragno (spider)
- scoiattolo (squirrel)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the CNN on the dataset in `archive/raw-img/`:

```bash
python train.py --data_dir archive/raw-img --epochs 50 --batch_size 32 --lr 0.001
```

**Arguments:**
- `--data_dir`: Path to raw-img directory (default: `archive/raw-img`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save model checkpoints (default: `./checkpoints`)
- `--log_dir`: Directory to save training logs and plots (default: `./logs`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

The training script will:
- Split the data into train/validation/test sets (80%/10%/10%)
- Apply data augmentation to training images
- Save the best model based on validation accuracy
- Save class mappings to `checkpoints/class_mappings.json`
- **Generate training curves and save them to `logs/` directory:**
  - `training_curves.png/pdf`: Combined loss and accuracy plots
  - `loss_curve.png/pdf`: Loss curves (training, validation, test)
  - `accuracy_curve.png/pdf`: Accuracy curves (training, validation, test)
  - `training_history.json`: Complete training history with all metrics
- Evaluate on test set and include results in plots

### Testing/Classifying an Image

Classify a single image using the trained model:

```bash
python test_image.py --image path/to/image.jpg --model checkpoints/best_model.pth
```

**Arguments:**
- `--image`: Path to the image file to classify (required)
- `--model`: Path to trained model checkpoint (default: `./checkpoints/best_model.pth`)
- `--mappings`: Path to class mappings JSON (default: `./checkpoints/class_mappings.json`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--top_k`: Number of top predictions to display (default: 5)

The script will:
- Load and preprocess the image
- Run inference through the CNN
- Apply softmax to get probabilities for all classes
- Display the predicted class and confidence
- Show top-k predictions and all class probabilities

## Model Architecture

The CNN consists of:
- **4 Convolutional Blocks**: Each with 2 conv layers, batch normalization, ReLU activation, max pooling, and dropout
- **3 Fully Connected Layers**: With dropout for regularization
- **Output Layer**: 10 classes (one for each animal type)

The model uses:
- Batch normalization for stable training
- Dropout for regularization
- Data augmentation (random flips, rotations, color jitter)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## How It Works

1. **Training**: The model learns to classify images by minimizing cross-entropy loss
2. **Inference**: 
   - Input image is preprocessed (resized to 224x224, normalized)
   - Forward pass through the CNN produces logits
   - Softmax is applied to convert logits to probabilities
   - The class with the highest probability is selected as the prediction

## Training Artifacts

After training, the `logs/` directory will contain:

- **`training_curves.png/pdf`**: Combined visualization of loss and accuracy curves
- **`loss_curve.png/pdf`**: High-resolution loss curves (training, validation, test)
- **`accuracy_curve.png/pdf`**: High-resolution accuracy curves (training, validation, test)
- **`training_history.json`**: Complete training history including:
  - Loss and accuracy for each epoch
  - Best validation accuracy
  - Test set results
  - Training hyperparameters
  - Timestamp

These artifacts are suitable for inclusion in research papers and presentations.

## Files

- `model.py`: CNN architecture definition
- `data_loader.py`: Dataset and data loading utilities
- `train.py`: Training script with logging and plotting
- `test_image.py`: Inference/classification script
- `requirements.txt`: Python dependencies
- `archive/translate.py`: Class name translations (Italian ↔ English)

