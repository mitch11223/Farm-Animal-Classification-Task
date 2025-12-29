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


##Make sure you have the proper 'archive' folder from: 'https://www.kaggle.com/datasets/alessiocorrado99/animals10?select=raw-img' in the same wd as the scripts.
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
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

The training script will:
- Split the data into train/validation/test sets (80%/10%/10%)
- Apply data augmentation to training images
- Save the best model based on validation accuracy
- Save class mappings to `checkpoints/class_mappings.json`

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

## Files

- `model.py`: CNN architecture definition
- `data_loader.py`: Dataset and data loading utilities
- `train.py`: Training script
- `test_image.py`: Inference/classification script
- `requirements.txt`: Python dependencies
- `archive/translate.py`: Class name translations (Italian ↔ English)

