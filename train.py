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


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', 
                learning_rate=0.001, save_dir='./checkpoints'):
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
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss and optimizer
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
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False, unit="batch")
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            batch_loss = loss.item()
            train_loss += batch_loss
            batch_count += 1
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar with current batch loss and accuracy
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Progress bar for validation batches
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
                
                # Update progress bar with current batch loss and accuracy
                batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                val_pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update epoch progress bar with summary
        epoch_info = f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        epoch_info += f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
        epoch_pbar.set_postfix_str(epoch_info)
        
        # Save best model
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
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = get_data_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Save class mappings
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'class_mappings.json'), 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }, f, indent=2)
    
    # Create model
    num_classes = len(class_to_idx)
    model = AnimalClassifierCNN(num_classes=num_classes).to(device)
    
    print(f"\nModel created with {num_classes} output classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
    
    print("\nTraining finished!")


if __name__ == '__main__':
    main()

