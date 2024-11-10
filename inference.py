import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
import os

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations (should match those used during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load class names from a JSON file saved during training
try:
    with open('class_names.json', 'r') as f:
        class_data = json.load(f)
    class_names = class_data["classes"]
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes.")
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading class names: {e}")
    sys.exit(1)

# Load the saved model
model_save_path = "food_classifier_model.pth"
try:
    # Define the model architecture (must match the model used during training)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_save_path}")
    sys.exit(1)

# Inference function with probabilities for all classes
def predict_image(image_path):
    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return None

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]  # Apply softmax to get probabilities
        # Print probabilities for each class
        for i, prob in enumerate(probabilities):
            print(f"Class: {class_names[i]}, Probability: {prob.item() * 100:.2f}%")
        # Return the most confident class and its probability
        _, predicted = torch.max(output, 1)
        class_name = class_names[predicted.item()]
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

