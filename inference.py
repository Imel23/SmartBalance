import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import json
import sys
import os

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations (should match those used during validation and testing)
transform_val_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved model
model_save_path = "food_classifier_model.pth"
try:
    # Define the model architecture (must match the model used during training)
    model = models.resnet50(weights=None)
    num_classes = 21  # Update this if the number of classes changes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_save_path}")
    sys.exit(1)

# Load class_to_idx mapping from the training dataset
train_dir = './split/train'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_val_test)
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
sorted_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(sorted_class_names)
print(f"Loaded {num_classes} classes.")

# Inference function with probabilities for all classes
def predict_image(image_path):
    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return None, None

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform_val_test(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]  # Apply softmax to get probabilities
        # Print probabilities for each class
        for i, prob in enumerate(probabilities):
            print(f"Class: {sorted_class_names[i]}, Probability: {prob.item() * 100:.2f}%")
        # Return the most confident class and its probability
        _, predicted = torch.max(output, 1)
        class_name = sorted_class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item() * 100
        return class_name, confidence

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
    else:
        image_path = sys.argv[1]
        predicted_class, confidence = predict_image(image_path)
        if predicted_class is not None:
            print(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f}%)")

