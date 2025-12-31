import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import argparse
from model import AnimalClassifierCNN
from data_loader import get_data_loaders
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#this script is used to train the model on the dataset, and save the model and the training curves, the test results and the class mappings



def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         test_loss=None, test_acc=None, save_dir='./logs'):
    """
    Plot and save training/validation/test curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        test_loss: Test loss (optional)
        test_acc: Test accuracy (optional)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    #Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    #Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    if test_loss is not None:
        ax1.axhline(y=test_loss, color='g', linestyle='--', 
                   label=f'Test Loss ({test_loss:.4f})', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    #Plot 2: Accuracy curves
    ax2 = axes[1]
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    if test_acc is not None:
        ax2.axhline(y=test_acc, color='g', linestyle='--', 
                   label=f'Test Accuracy ({test_acc:.2f}%)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    #Save combined plot
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_curves.pdf'), bbox_inches='tight')
    plt.close()
    
    #Create separate plots for better paper quality
    #Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    if test_loss is not None:
        plt.axhline(y=test_loss, color='g', linestyle='--', 
                   label=f'Test Loss ({test_loss:.4f})', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'loss_curve.pdf'), bbox_inches='tight')
    plt.close()
    
    #Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    if test_acc is not None:
        plt.axhline(y=test_acc, color='g', linestyle='--', 
                   label=f'Test Accuracy ({test_acc:.2f}%)', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir}/")



#trained on my cpu
def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        test_loss, test_acc
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100 * test_correct / test_total
    
    return test_loss, test_acc
def train_model(model, train_loader, val_loader, num_epochs=50, device='cpu', 
                learning_rate=0.001, save_dir='./checkpoints', log_dir='./logs'):
    """
    Train the CNN model.
    
    Args:
        model: The CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        learning_rate: Learning rate
        save_dir: Directory to save model checkpoints
    """
    
    #Create save directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 60)
    
    #Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        #Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        #Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False, unit="batch")
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            #Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            #Backward pass
            loss.backward()
            optimizer.step()
            
            #Statistics
            batch_loss = loss.item()
            train_loss += batch_loss
            batch_count += 1
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            #Update progress bar with current batch loss and accuracy
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            avg_loss = train_loss / batch_count
            avg_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'acc': f'{batch_acc:.2f}%',
                'avg_acc': f'{avg_acc:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        #Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        #Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                       leave=False, unit="batch")
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                batch_loss = loss.item()
                val_loss += batch_loss
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                #Update progress bar with current batch loss and accuracy
                batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                val_pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        #Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        #Save statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        #Update epoch progress bar with summary
        epoch_info = f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        epoch_info += f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
        epoch_pbar.set_postfix_str(epoch_info)
        
        #Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            epoch_pbar.write(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    #Save training history to JSON
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'timestamp': datetime.now().isoformat()
    }
    
    history_path = os.path.join(log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return train_losses, val_losses, train_accs, val_accs


def main():
    parser = argparse.ArgumentParser(description='Train Animal Classifier CNN')
    parser.add_argument('--data_dir', type=str, default='archive/raw-img',
                       help='Path to raw-img directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save training logs and plots')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training (default: cpu)')
    
    args = parser.parse_args()
    
    # Set device - use CUDA only if explicitly requested and available
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if args.device == 'cuda':
            print("Warning: CUDA requested but not available. Using CPU instead.")
    
    #Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = get_data_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    #Save class mappings
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'class_mappings.json'), 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }, f, indent=2)
    
    #Create model
    num_classes = len(class_to_idx)
    model = AnimalClassifierCNN(num_classes=num_classes).to(device)
    
    print(f"\nModel created with {num_classes} output classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    #Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    #Evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    
    #Generate and save training curves
    print("\nGenerating training curves...")
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        test_loss=test_loss, test_acc=test_acc,
        save_dir=args.log_dir
    )
    
    #Update training history with test results
    history_path = os.path.join(args.log_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            training_history = json.load(f)
        training_history['test_loss'] = test_loss
        training_history['test_acc'] = test_acc
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    print("\nTraining finished!")
    print(f"All artifacts saved to {args.log_dir}/")


if __name__ == '__main__':
    main()

