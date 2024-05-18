import torch
import torch.nn as nn

from data import load_data, visualise_data


# Set the random seed for manual reproductibility
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device used: {device}")

# Load data
data = load_data()
visualise_data(data)
