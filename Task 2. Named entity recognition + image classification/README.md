# Animal Classification and Detection Project

This project combines Named Entity Recognition (NER) and Image Classification (CNN) to detect and verify animal mentions in text and images.

## Project Structure

```
project/
├── CNN_realization/
│   ├── translate.py       # Utility for translating folder from italian to english
│   ├── CNN_data/
│   │   └── raw-img/      # Animal images organized by class folders
│   │       ├── butterfly/
│   │       ├── cat/
│   │       ├── chicken/
│   │       ├── cow/
│   │       ├── dog/
│   │       ├── elephant/
│   │       ├── horse/
│   │       ├── scoiattolo/
│   │       ├── sheep/
│   │       └── spider/
│   ├── CNN_model/
│   │   ├── animal_classifier.pth     # Trained CNN model
│   │   ├── classification_report.txt  # Model performance report
│   │   ├── confusion_matrix.png      # Confusion matrix visualization
│   │   └── training_curves.png       # Training metrics visualization
│   ├── CNN_data_analysis.ipynb       # Data analysis notebook
│   ├── CNN_inference.py              # Script for running inference
│   └── CNN_train.py                  # Script for training the CNN model
│
├── NER_realization/
│   ├── NER_data/
│   │   └── NER_Dataset.csv           # Dataset for NER training
│   ├── NER_model/                    # Trained NER model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── NER_data_analysis.ipynb       # Data analysis notebook
│   ├── NER_inference.py              # Script for running inference
│   └── NER_train.py                  # Script for training the NER model
│
├── demo.ipynb                        # Demo notebook (this file)
├── test.py                           # Combined pipeline testing script
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd animal-classification-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download Dataset
To download the **Animals-10** dataset, run the following command:
   ```bash
   cd CNN_data
   kaggle datasets download -d alessiocorrado99/animals10
   unzip animals10.zip -d raw-img
   ```
4. Prepare Dataset
Extract the dataset and rename class folders using:
   ```bash
   python translate.py
   ```
## Models

### 1. CNN Image Classification Model

- **Architecture**: ResNet18 (pretrained on ImageNet)
- **Input**: 224x224 RGB images
- **Output**: Animal classification among 10 classes
- **Classes**: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, squirrel, spider

### 2. NER Model

- **Architecture**: BERT-based token classification
- **Input**: Text
- **Output**: Animal entity extraction
- **Entity Types**: B-ANIMAL (Beginning of animal entity)
### 3. Download

Download models from google disk: https://drive.google.com/drive/folders/1IZCtTTxHP8woyafc-LxPG3V2hyeUXgfc?usp=sharing

## Usage

### Interactive Demo

Run the demo notebook for an interactive experience:
```bash
jupyter notebook demo.ipynb
```

### Command Line Interface

Run the combined pipeline:
```bash
python test.py --ner_model_path NER_realization/NER_model --cnn_model_path CNN_realization/CNN_model/animal_classifier.pth --text "There is a dog in the picture" --image_path CNN_realization/CNN_data/raw-img/dog/1.jpeg
```

### Individual Components

1. Image Classification:
   ```bash
   python CNN_inference.py --model_path CNN_model/animal_classifier.pth --image_path CNN_data/raw-img/cat/1.jpeg
   ```

2. Named Entity Recognition:
   ```bash
   python NER_inference.py
   # Then enter a sentence when prompted, e.g., "I saw a cat and a dog yesterday"
   ```

## Training

### CNN Model Training

```bash
python CNN_train.py --data_dir CNN_data/raw-img --epochs 10 --batch_size 32
```

### NER Model Training

```bash
python NER_train.py
```

## Dataset

- **Image Dataset**: 10 animal classes with multiple images per class
- **NER Dataset**: Text samples with animal entity annotations

## Performance

Check the model performance in:
- CNN_model/classification_report.txt
- CNN_model/confusion_matrix.png
- CNN_model/training_curves.png

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- transformers
- Pillow
- scikit-learn
- matplotlib
- pandas
- numpy

## License

This project is licensed under the MIT License - see the LICENSE file for details.
