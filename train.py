import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from model import CNNClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# PATHS
TRAIN_PATH = r'D:\Real-vs-AI-Image-Detector-main\datasets\train'
TEST_PATH = r'D:\Real-vs-AI-Image-Detector-main\datasets\test'
MODEL_SAVE_PATH = "models/ai_real_classifier.pth"

class ImageClassifier:
    def __init__(self, train_path, test_path, val_split=0.2):
        self.train_path = train_path
        self.test_path = test_path
        self.val_split = val_split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Simpler transformations for faster processing
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Initialize model
        self.model = CNNClassifier(num_classes=2, input_size=32).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def load_datasets(self):
        """Load datasets with reduced size for faster training"""
        print("Loading datasets...")
        
        try:
            # Load training dataset
            full_train_dataset = datasets.ImageFolder(root=self.train_path, transform=self.transform)
            
            # Use only 20% of data for training (16,000 images instead of 80,000)
            # This will make training 5x faster
            train_size = int(0.2 * len(full_train_dataset))
            remaining_size = len(full_train_dataset) - train_size
            train_subset, _ = torch.utils.data.random_split(
                full_train_dataset, [train_size, remaining_size]
            )
            
            # Split into train and validation
            train_size = int((1 - self.val_split) * len(train_subset))
            val_size = len(train_subset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_subset, [train_size, val_size]
            )
            
            # Load test dataset (use full test set or subset)
            full_test_dataset = datasets.ImageFolder(root=self.test_path, transform=self.val_transform)
            # Use 20% of test data for faster evaluation
            test_size = int(0.2 * len(full_test_dataset))
            test_dataset, _ = torch.utils.data.random_split(
                full_test_dataset, [test_size, len(full_test_dataset) - test_size]
            )
            
            # Larger batch size for faster training
            batch_size = 128  # Increased from 64 to 128
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            self.class_names = full_train_dataset.classes
            print(f"Classes found: {self.class_names}")
            print(f"✅ Train samples (20%): {len(train_dataset)}")
            print(f"✅ Validation samples: {len(val_dataset)}")
            print(f"✅ Test samples (20%): {len(test_dataset)}")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return None, None, None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def train(self, num_epochs=10):  # Reduced from 30 to 10 epochs
        """Main training loop"""
        train_loader, val_loader, test_loader = self.load_datasets()
        
        if train_loader is None:
            return
        
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0.0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Batch size: {train_loader.batch_size}, Approx time per epoch: {len(train_loader) * 0.1:.0f} seconds")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            val_loss, val_acc = self.validate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(MODEL_SAVE_PATH)
                print(f"✅ Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # Test with full test set at the end
        print("\n" + "="*50)
        print("Testing on full test set...")
        full_test_dataset = datasets.ImageFolder(root=self.test_path, transform=self.val_transform)
        full_test_loader = DataLoader(full_test_dataset, batch_size=128, shuffle=False, num_workers=0)
        self.test(full_test_loader)
        
        print("\n✅ Training completed!")
    
    def test(self, test_loader):
        """Test the model"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100. * correct / total
        print(f"✅ Test Accuracy: {test_acc:.2f}%")
        return test_acc
    
    def save_model(self, path):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        class_names_for_app = []
        for name in self.class_names:
            if name.lower() == 'fake':
                class_names_for_app.append('ai')
            elif name.lower() == 'real':
                class_names_for_app.append('real')
            else:
                class_names_for_app.append(name)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_names': class_names_for_app,
        }, path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    classifier = ImageClassifier(TRAIN_PATH, TEST_PATH, val_split=0.2)
    classifier.train(num_epochs=10)  # Only 10 epochs instead of 30