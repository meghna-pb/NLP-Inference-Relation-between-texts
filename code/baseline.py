import torch
from data import load_data, visualise_data, tokenize_data, prepare_dataloaders
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer
import torch.nn as nn
import torch.optim as optim
from LSTMModel import LSTMModel
from GloVeModel import GloVeModel  # Import the GloVeModel class


# Load the data
data = load_data()
visualise_data(data)

# Prepare data for classical models
def prepare_data_for_classic_models(dataset):
    # Combine premise and hypothesis into a single text
    def combine_texts(examples):
        return {'text': examples['premise'] + " " + examples['hypothesis']}
    
    # Apply the combination function to the dataset
    dataset = dataset.map(combine_texts, remove_columns=['premise', 'hypothesis'])
    texts = dataset['train']['text']
    labels = dataset['train']['label']
    return texts, labels

texts, labels = prepare_data_for_classic_models(data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=272)

# Create a bag-of-words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_val_bow = vectorizer.transform(X_val)

# Classic models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
}

# Train and evaluate classical models
results = {}
for name, model in models.items():
    model.fit(X_train_bow, y_train)
    y_pred = model.predict(X_val_bow)
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")



# Train and evaluate the GloVe model
print("Training the GloVe model")
vocab = vectorizer.get_feature_names_out()
cooccurrence_matrix = (X_train_bow.T @ X_train_bow).tocoo()
glove_model = GloVeModel(n_dim=50, learning_rate=0.05, vocab=vocab, max_epochs=10, x_max=100, alpha=0.75)
glove_model.fit(cooccurrence_matrix)

# Use the learned word vectors to train a simple classifier (Logistic Regression)
glove_embeddings = glove_model.W
X_train_glove = X_train_bow @ glove_embeddings
X_val_glove = X_val_bow @ glove_embeddings

log_reg_glove = LogisticRegression(max_iter=1000)
log_reg_glove.fit(X_train_glove, y_train)
y_pred_glove = log_reg_glove.predict(X_val_glove)
glove_accuracy = accuracy_score(y_val, y_pred_glove)
results['GloVe + Logistic Regression'] = glove_accuracy
print(f"GloVe + Logistic Regression Accuracy: {glove_accuracy * 100:.2f}%")

# Prepare data for the LSTM model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_datasets = tokenize_data(data)
train_loader, val_loader = prepare_dataloaders(tokenized_datasets)

# Parameters for the LSTM model
input_dim = len(tokenizer.vocab)
hidden_dim = 256
output_dim = 3  # number of labels
n_layers = 2

# Initialize and train the LSTM model
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, n_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
num_epochs = 3
print(len(train_loader))
for epoch in range(num_epochs):
    lstm_model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Evaluate the LSTM model
lstm_model.eval()
total_eval_accuracy = 0
total_eval_items = 0
for batch in val_loader:
    inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
    with torch.no_grad():
        outputs = lstm_model(inputs)
    predictions = torch.argmax(outputs, dim=1)
    total_eval_accuracy += (predictions == labels).sum().item()
    total_eval_items += len(labels)

lstm_accuracy = total_eval_accuracy / total_eval_items
results['LSTM'] = lstm_accuracy
print(f"LSTM Model Accuracy: {lstm_accuracy * 100:.2f}%")

# Compare performances
print("Performance comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
