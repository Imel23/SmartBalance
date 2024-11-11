import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2  # OpenCV for webcam access
import json
import numpy as np

# Device configuration (use CPU, suitable for Raspberry Pi)
device = torch.device("cpu")
print(f"Using device: {device}")

# Define data transformations (match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load class names from a JSON file
try:
    with open('class_names.json', 'r') as f:
        class_data = json.load(f)
    class_names = class_data["classes"]
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes.")
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading class names: {e}")
    exit(1)

# Load the trained model
model_save_path = "food_classifier_model.pth"
try:
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_save_path}")
    exit(1)

# Inference function for webcam frames
def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
        class_name = class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item() * 100
        return class_name, confidence

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Run inference on webcam frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform prediction
        class_name, confidence = predict_frame(frame)
        
        # Display the prediction on the frame
        label = f"{class_name} ({confidence:.2f}%)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("Webcam Inference", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
