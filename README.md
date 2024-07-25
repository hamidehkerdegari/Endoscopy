# Endoscopy Classification Project

This project aims to perform multiclass classification on endoscopy images using PyTorch.

## Project Structure

```plaintext
endoscopy_classification/
├── data/                     # Empty directory for data files
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Custom dataset class
│   │   ├── transforms.py     # Data augmentation and preprocessing
│   │   └── dataloader.py     # DataLoader wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py          # Model architecture
│   │   └── loss.py           # Custom loss functions if any
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py          # Training loop
│   │   ├── evaluate.py       # Evaluation logic
│   │   └── utils.py          # Utility functions
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration settings
│   │   └── paths.py          # File paths and directory settings
│   └── main.py               # Entry point for running the project
├── scripts/
│   ├── train_model.py        # Script for training the model
│   └── evaluate_model.py     # Script for evaluating the model
├── tests/
│   ├── test_data.py          # Unit tests for data-related modules
│   ├── test_model.py         # Unit tests for model-related modules
│   ├── test_training.py      # Unit tests for training-related modules
│   └── test_utils.py         # Unit tests for utility functions
├── .gitignore                # Git ignore file
├── README.md                 # Project description and instructions
├── requirements.txt          # Python dependencies
└── setup.py                  # Setup script for the project
```

## Setup Instructions

1. Clone the repository:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd endoscopy_classification
```

3. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the model

```bash
python scripts/train_model.py
```

### Evaluating the model

```bash
python scripts/evaluate_model.py
```

## Project Description

This project involves training a deep learning model to classify endoscopy images into three categories: intestinal metaplasia (IM), gastritis atrophy (GA), and normal. The dataset includes images from five gastric views: Gastric Antrum, Gastric Angle, Cardia, Gastric Fundus, and Gastric Body. The project is designed to handle data preprocessing on the fly using PyTorch's data pipeline capabilities.
