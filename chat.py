from transformers import BertTokenizer, BertForMaskedLM, TextGenerationPipeline
import torch

def load_model(model_path):
    """
    Load a pre-trained BERT model and its tokenizer from a specified path.

    Args:
        model_path (str): The path to the directory where the model and tokenizer are saved.

    Returns:
        tuple: A tuple containing the model and tokenizer instances.
    """
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_text(prompt, model, tokenizer):
    """
    Generate text based on a given prompt using BERT model.

    Args:
        prompt (str): The text prompt to generate text from.
        model: The pre-trained BERT model.
        tokenizer: The tokenizer for the BERT model.

    Returns:
        str: The generated text.
    """
    model.eval()  # Set model to evaluation mode
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')  # Encode the prompt text
    
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True)  # Generate text
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)  # Decode generated text
    return generated_text

model_path = "./results"  # Path to your model directory

model, tokenizer = load_model(model_path)  # Load the model and tokenizer

def answer_question(question):
    """
    Generate and print an answer to a given question.

    Args:
        question (str): The question to generate an answer for.
    """
    generated_text = generate_text(question, model, tokenizer)
    print(f"\n>> {question}\n{generated_text}\n")

# Example usage
answer_question("The future of AI in society is")
answer_question("Co je statutarny audit?")
answer_question("What is statutory audit?")
