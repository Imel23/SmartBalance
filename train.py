import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import json
import numpy as np
from sklearn.metrics import classification_report
from torch.cuda.amp import GradScaler, autocast

# Verify CUDA availability
print("Is CUDA available?", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please install PyTorch with CUDA support.")

# Device configuration (use GPU if available)
device = torch.device("cuda")
print(f"Training on: {device}")

# Define data transformations with data augmentation for training
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transformations for validation and test (no augmentation)
transform_val_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dir = './split/train'
val_dir = './split/val'
test_dir = './split/test'

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val_test)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# Map class indices to class names using the dataset's class_to_idx
# Ensure the class names from the JSON match the folder names
# Load class names from JSON
with open('class_names.json', 'r') as f:
    class_data = json.load(f)
json_class_names = class_data['classes']

# Get the class_to_idx mapping from the training dataset
class_to_idx = train_dataset.class_to_idx

# Create idx_to_class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Verify that class names in JSON match the dataset classes
dataset_classes = [idx_to_class[i] for i in range(len(idx_to_class))]
if set(json_class_names) != set(dataset_classes):
    raise ValueError("Class names in JSON do not match dataset class names.")

# Sort json_class_names based on dataset_classes order
# We need to map the indices correctly
sorted_class_names = [None] * len(json_class_names)
for idx in range(len(json_class_names)):
    class_name = idx_to_class[idx]
    sorted_class_names[idx] = class_name

# Now, 'sorted_class_names' should be in the same order as class indices used by the datasets
num_classes = len(sorted_class_names)

# Define the model (using a pre-trained ResNet50 and modifying the final layer)
model = models.resnet50(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)  # Move model to GPU

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training function with early stopping and model checkpointing
def train_model(num_epochs, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = GradScaler()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        # Evaluate on validation set
        val_acc, val_report, val_loss = evaluate_model(val_loader)
        # Scheduler step
        scheduler.step(val_loss)
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% (improved)")
        else:
            patience_counter += 1
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% (no improvement)")
            if patience_counter >= patience:
                print("Early stopping")
                break

# Evaluation function with additional metrics
def evaluate_model(loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100 * correct / total
    avg_loss = total_loss / len(loader)
    # Calculate classification report
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=sorted_class_names,
        output_dict=True
    )
    return acc, class_report, avg_loss

# Run training
num_epochs = 100
patience = 10
train_model(num_epochs, patience)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test data
test_acc, test_report, test_loss = evaluate_model(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
print("Classification Report:")
print(classification_report(
    test_dataset.targets,
    [p for p in np.concatenate([predicted.cpu().numpy() for images, _ in test_loader for predicted in [torch.max(model(images.to(device)), 1)[1]]])],
    target_names=sorted_class_names
))

# Save the model's state dictionary
model_save_path = "food_classifier_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Inference function
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform_val_test(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_name = sorted_class_names[predicted.item()]
        return class_name

# Example of using inference
image_path = 'test.png'
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")

