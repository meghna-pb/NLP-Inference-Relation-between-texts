from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader

def load_data():
    """Load the SNLI dataset and filter out entries with label -1.

    Returns:
        datatset (datasets.DatasetDict): A dictionary containing the 'train', 'validation', 
                                        and 'test' splits of the SNLI dataset, with entries
                                        where label is not -1.
    """
    dataset = load_dataset("snli")
    dataset = dataset.filter(lambda example: example['label'] != -1) # filter out pairs with no label
    return dataset

def tokenize_data(dataset, tokenizer_name='distilbert-base-uncased', max_length=128):
    """Tokenize the dataset using the specified tokenizer.

    Args:
        dataset (datasets.DatasetDict): The dataset to tokenize.
        tokenizer_name (str): The name of the tokenizer to use.
        max_length (int): The maximum length of the tokenized sequences.

    Returns:
        tokenized_datasets (datasets.DatasetDict): The tokenized dataset.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_datasets

def prepare_dataloaders(tokenized_datasets, batch_size=8):
    """Prepare dataloaders for the tokenized dataset.

    Args:
        tokenized_datasets (datasets.DatasetDict): The tokenized dataset.
        batch_size (int): The batch size for the dataloaders.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
    """
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def visualise_data(dataset) -> None:
    """
    Visualize various statistics and examples from the SNLI dataset.

    Parameters:
        dataset (datasets.DatasetDict): The dataset to analyze, expected to contain 
                                        'train', 'validation', and 'test' splits.
    
    Prints:
        - The number of pairs in each dataset split.
        - The average length of premises and hypotheses in the train split.
        - The count of each label (Entailment, Neutral, Contradiction) in the train split.
        - An example premise and hypothesis for each label type.
    """

    # Dataset caracteristics
    print("\nNumber of pairs in each dataset (train, validation and test):")
    for split in dataset.keys():
        formatted_number = f"{len(dataset[split]):,}".replace(",", " ")
        print(f"Number of pairs in {split}: {formatted_number}")

    # Train dataset caracteristics
    print("\nTrain dataset caracteristics:")

    # Average length of premises and hypotheses
    premise_lengths = dataset['train'].map(lambda x: {'length': len(x['premise'].split())}, remove_columns=['hypothesis', 'label'])
    hypothesis_lengths = dataset['train'].map(lambda x: {'length': len(x['hypothesis'].split())}, remove_columns=['premise', 'label'])
    avg_premise_length = sum(premise_lengths['length']) / len(premise_lengths)
    avg_hypothesis_length = sum(hypothesis_lengths['length']) / len(hypothesis_lengths)
    print(f"Average length of premises: {np.round(avg_premise_length,2)} words")
    print(f"Average length of hypotheses: {np.round(avg_hypothesis_length,2)} words")

    # Label distribution
    print("\nLabel distribution:")
    labels = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}
    
    label_counts = {0: 0, 1: 0, 2: 0}
    for example in dataset['train']:
        label_counts[example['label']] += 1

    for idx, label in labels.items():
        count_formatted = f"{(label_counts[idx]):,}".replace(",", " ")
        print(f"{label} count: {count_formatted}")

    # Pair example with label
    print("\nExample for each label type:")
    
    examples = {}
    for label_id, label_name in labels.items():
        example = dataset['train'].filter(lambda example: example['label'] == label_id, load_from_cache_file=False)
        if example.num_rows > 0:
            examples[label_name] = example[0]

    for label, example in examples.items():
        print(f"Label: {label}")
        print(f"Premise: {example['premise']}")
        print(f"Hypothesis: {example['hypothesis']}\n")