import torch
import torch.nn as nn
from data import load_data, visualise_data, tokenize_data, prepare_dataloaders
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification
from tqdm import tqdm


# Set the random seed for manual reproductibility
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device used: {device}")

# Load data
data = load_data()
# visualise_data(data)

tokenized_datasets = tokenize_data(data)
train_loader, val_loader = prepare_dataloaders(tokenized_datasets)


# Setting up the pre-trained model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training the model
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch in progress_bar:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix(loss=loss.item())
    
    # Print average loss for the epoch
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader)}")


# Testing the model
model.eval()
total_eval_accuracy = 0

for batch in val_loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    total_eval_accuracy += (predictions == batch['labels']).sum().item()

print(f"Validation Accuracy: {total_eval_accuracy / len(val_loader)}")

# Launch baseline comparison