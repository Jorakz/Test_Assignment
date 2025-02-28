import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torchvision import transforms, models
from PIL import Image
import argparse
import os


# Load NER Model
def load_ner_model(model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    model.eval()
    id2tag = {0: "O", 1: "B-ANIMAL"}  # Label mapping
    return tokenizer, model, id2tag


def ner_predict(text, tokenizer, model, id2tag):
    """Extracts animal entities from text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predictions = outputs.logits.argmax(dim=2).squeeze().tolist()
    predicted_labels = [id2tag[pred] for pred in predictions]
    entities = []
    current_entity = ""
    for i in range(len(tokens)):
        if tokens[i] in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if predicted_labels[i] == "B-ANIMAL":
            if tokens[i].startswith("##"):
                current_entity += tokens[i][2:]
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = tokens[i]
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = ""
    if current_entity:
        entities.append(current_entity)
    return entities


# Load CNN Model
def load_cnn_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, class_names


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def cnn_predict(image_path, model, class_names, device):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]


if __name__ == "__main__":
    text = ''
    parser = argparse.ArgumentParser(description='Full pipeline: NER + Image Classification')
    parser.add_argument('--ner_model_path', type=str, default='NER_realization/NER_model')
    parser.add_argument('--cnn_model_path', type=str, default='CNN_realization/CNN_model/animal_classifier.pth')
    parser.add_argument('--text', type=str, default='There is a lion in the picture')
    parser.add_argument('--image_path', type=str, default='CNN_realization/CNN_data/raw-img/cat/1.jpeg')

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    tokenizer, ner_model, id2tag = load_ner_model(args.ner_model_path)
    cnn_model, class_names = load_cnn_model(args.cnn_model_path, device)

    # Get predictions
    extracted_animals = ner_predict(args.text, tokenizer, ner_model, id2tag)
    predicted_class = cnn_predict(args.image_path, cnn_model, class_names, device)

    # Compare predictions
    match = predicted_class in extracted_animals
    print(f"NER Extracted Animals: {extracted_animals}")
    print(f"CNN Predicted Class: {predicted_class}")
    print(f"Match: {match}")
