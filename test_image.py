import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import json
import os
from model import AnimalClassifierCNN


def load_model(model_path, num_classes, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = AnimalClassifierCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path, device='cuda'):
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to image file
        device: Device to load image on
    
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def classify_image(model, image_tensor, idx_to_class, top_k=5):
    """
    Classify an image and return probabilities for all classes.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        idx_to_class: Mapping from class index to class name
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and probabilities
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(idx_to_class)), dim=1)
        
        # Convert to lists
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()
        
        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        
        # Create results dictionary
        results = {
            'top_predictions': [],
            'all_probabilities': {}
        }
        
        # Add top-k predictions
        for prob, idx in zip(top_probs, top_indices):
            class_name = idx_to_class[idx]
            results['top_predictions'].append({
                'class': class_name,
                'probability': float(prob),
                'percentage': float(prob * 100)
            })
        
        # Add all probabilities
        for idx, prob in enumerate(all_probs):
            class_name = idx_to_class[idx]
            results['all_probabilities'][class_name] = {
                'probability': float(prob),
                'percentage': float(prob * 100)
            }
        
        # Get predicted class (highest probability)
        predicted_idx = top_indices[0]
        predicted_class = idx_to_class[predicted_idx]
        predicted_prob = top_probs[0]
        
        results['predicted_class'] = predicted_class
        results['predicted_probability'] = float(predicted_prob)
        results['predicted_percentage'] = float(predicted_prob * 100)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Test/Classify an image using trained CNN')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file to classify')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--mappings', type=str, default='./checkpoints/class_mappings.json',
                       help='Path to class mappings JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to display')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load class mappings
    if not os.path.exists(args.mappings):
        raise FileNotFoundError(f"Class mappings file not found: {args.mappings}")
    
    with open(args.mappings, 'r') as f:
        mappings = json.load(f)
    
    idx_to_class = {int(k): v for k, v in mappings['idx_to_class'].items()}
    num_classes = len(idx_to_class)
    
    print(f"Loaded {num_classes} classes")
    print(f"Classes: {list(mappings['class_to_idx'].keys())}")
    
    # Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model, num_classes, device)
    print("Model loaded successfully!")
    
    # Preprocess image
    print(f"\nLoading and preprocessing image: {args.image}...")
    image_tensor = preprocess_image(args.image, device)
    
    # Classify image
    print("Classifying image...")
    results = classify_image(model, image_tensor, idx_to_class, top_k=args.top_k)
    
    # Display results
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"\nPredicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['predicted_percentage']:.2f}%")
    print(f"\nTop {args.top_k} Predictions:")
    print("-" * 60)
    for i, pred in enumerate(results['top_predictions'], 1):
        print(f"{i}. {pred['class']:15s} - {pred['percentage']:6.2f}%")
    
    print("\n" + "=" * 60)
    print("All Class Probabilities:")
    print("-" * 60)
    # Sort by probability
    sorted_probs = sorted(
        results['all_probabilities'].items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    )
    for class_name, prob_info in sorted_probs:
        print(f"{class_name:15s} - {prob_info['percentage']:6.2f}%")
    
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()

