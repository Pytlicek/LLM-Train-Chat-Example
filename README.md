# BERT-Based Text Generation

This repository contains code for training a BERT model for masked language modeling and generating text based on prompts using the trained model.

## Installation

Before running the code, ensure you have Python and PyTorch installed. You also need to install the `transformers` library by Hugging Face:

```
pip install transformers
pip install tokenizers
pip install torch
```

## Files

- `train.py`: This script trains a BERT model on a text dataset for masked language modeling. It uses the transformers library and a custom dataset class for training.
- `chat.py`: This script demonstrates how to generate text based on prompts using the trained BERT model. Note that BERT is not primarily designed for text generation, so the results might not always be coherent.

## Usage

### Training the Model

Run the `train.py` script to train the model. Ensure you have a dataset named `dataset.txt` in the same directory:

```
python train.py
```

The trained model and tokenizer will be saved in the `./results` directory.

### Generating Text

Use the `chat.py` script to generate text based on prompts using the trained model:

```
python chat.py
```

## Note

The generated text quality might vary as BERT is primarily designed for understanding tasks rather than generation. However, this project serves as a demonstration of custom training and text generation capabilities.

Enjoy exploring BERT-based text generation!
