import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import json

# Load custom class names from JSON
with open('class_names.json', 'r') as f:
    class_data = json.load(f)
class_names = class_data['classes']

# Verify CUDA availability
print("Is CUDA available?", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please install PyTorch with CUDA support.")

# Device configuration (use GPU if available)
device = torch.device("cuda")
print(f"Training on: {device}")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
train_dir = './split/train'
val_dir = './split/val'
test_dir = './split/test'

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# Define the model (using a pre-trained ResNet and modifying the final layer)
model = models.resnet18(pretrained=True)
num_classes = len(class_names)  # Use custom class count from JSON
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)  # Move model to GPU

# Verify model parameters are on GPU
for param in model.parameters():
    assert param.is_cuda, "Model parameter is not on GPU"

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # Move data to GPU

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation accuracy
        val_acc = evaluate_model(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")

# Evaluation function
def evaluate_model(loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # Move data to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Run training
num_epochs = 30
train_model(num_epochs)

# Evaluate on test data
test_accuracy = evaluate_model(test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save the model's state dictionary
model_save_path = "food_classifier_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Inference function
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        # Use custom class names loaded from JSON
        class_name = class_names[predicted.item()]
        return class_name

# Example of using inference
image_path = 'test.png'
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")

