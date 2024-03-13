from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import torch

def load_dataset(file_path):
    """
    Load text data from a specified file path.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        list[str]: A list of sentences from the dataset.
    """
    with open(file_path, 'r', encoding='utf8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def prepare_inputs(sentences, tokenizer):
    """
    Tokenize sentences and prepare inputs for the model.

    Args:
        sentences (list[str]): A list of sentences to tokenize.
        tokenizer: The tokenizer instance for tokenizing the sentences.

    Returns:
        dict: A dictionary of tokenized inputs including input ids and labels.
    """
    inputs = tokenizer(sentences, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
    inputs['labels'] = inputs.input_ids.detach().clone()  # For BertForMaskedLM, labels are the input_ids
    return inputs

class CustomDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for handling the encoding of text data.
    """
    def __init__(self, encodings):
        """
        Initializes the dataset with encodings.

        Args:
            encodings (dict): Encoded inputs from the tokenizer.
        """
        self.encodings = encodings

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        """
        Returns an item by index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary of tensor values for the specified index.
        """
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Load dataset and prepare inputs
    sentences = load_dataset("dataset.txt")
    inputs = prepare_inputs(sentences, tokenizer)
    
    # Initialize model
    model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",          # Directory where the model checkpoints will be saved
        num_train_epochs=2,              # Number of training epochs
        per_device_train_batch_size=2,   # Batch size per device during training
        logging_dir="./logs",            # Directory for storing logs
        save_strategy="epoch",           # Saving strategy
        logging_steps=10,                # Log metrics every X steps
    )
    
    # Initialize custom dataset
    train_dataset = CustomDataset(inputs)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Start training
    trainer.train()

    # Save model and tokenizer for later use
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')

if __name__ == "__main__":
    main()
