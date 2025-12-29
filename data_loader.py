import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob


class AnimalDataset(Dataset):
    """
    Custom dataset for loading animal images from archive/raw-img/
    """
    
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d))])
        
        # Create class_to_idx mapping if not provided
        if class_to_idx is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all images with their labels
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Support multiple image formats
            image_extensions = ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
            for ext in image_extensions:
                image_paths = glob.glob(os.path.join(class_path, ext))
                for img_path in image_paths:
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_dir, batch_size=32, train_split=0.8, val_split=0.1, test_split=0.1):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to archive/raw-img/ directory
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
    
    Returns:
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class
    """
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset to get class mappings (without transforms for now)
    full_dataset = AnimalDataset(data_dir, transform=None)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = full_dataset.idx_to_class
    
    # Split dataset indices
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    indices = list(range(dataset_size))
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        indices, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets with appropriate transforms
    train_dataset_full = AnimalDataset(data_dir, transform=train_transform, class_to_idx=class_to_idx)
    val_dataset_full = AnimalDataset(data_dir, transform=val_test_transform, class_to_idx=class_to_idx)
    test_dataset_full = AnimalDataset(data_dir, transform=val_test_transform, class_to_idx=class_to_idx)
    
    # Create subsets with the split indices
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class

