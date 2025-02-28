import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
import json


def load_model(model_path, device):
    """Load the trained model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']

    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image_path):
    """Preprocess the image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict(image_path, model, class_names, device):
    """Run inference on a single image and return the predicted class."""
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on an image using trained CNN')
    parser.add_argument('--model_path', type=str, default='CNN_model/animal_classifier.pth')
    parser.add_argument('--image_path', type=str, default='CNN_data/raw-img/cat/1.jpeg')

    args = parser.parse_args()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, class_names = load_model(args.model_path, device)

    # Predict
    prediction = predict(args.image_path, model, class_names, device)
    print(f"Predicted class: {prediction}")
