from datasets import load_dataset


def load_data():
    dataset = load_dataset("snli")
    dataset = dataset.filter(lambda example: example['label'] != -1)
    return dataset