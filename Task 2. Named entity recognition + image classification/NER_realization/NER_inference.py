import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# Load trained model and tokenizer
model_path = "NER_model"  # Ensure this is the correct path
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

model.eval()  # Set model to evaluation mode
id2tag = {0: "O", 1: "B-ANIMAL"}  # Label mapping

def predict(text):
    """
    Predicts and extracts only animal entities (B-ANIMAL) from the given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predictions = outputs.logits.argmax(dim=2).squeeze().tolist()
    predicted_labels = [id2tag[pred] for pred in predictions]

    # Extract only B-ANIMAL labels
    entities = []
    current_entity = ""

    for i in range(len(tokens)):
        if tokens[i] in ["[CLS]", "[SEP]", "[PAD]"]:
            continue  # Ignore special tokens

        if predicted_labels[i] == "B-ANIMAL":
            if tokens[i].startswith("##"):
                current_entity += tokens[i][2:]  # Merge subword token
            else:
                if current_entity:
                    entities.append(current_entity)  # Store previous entity
                current_entity = tokens[i]  # Start a new entity
        else:
            if current_entity:
                entities.append(current_entity)  # Store last detected entity
                current_entity = ""

    if current_entity:
        entities.append(current_entity)  # Ensure last detected entity is stored

    return entities


if __name__ == "__main__":
    text = input("Enter a sentence: ")
    extracted_animals = predict(text)

    if extracted_animals:
        print(f"\nExtracted animal entities: {extracted_animals}")
    else:
        print("\nNo animal entities found.")