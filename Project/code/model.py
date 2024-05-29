import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import random
import os
import torch.nn
import tqdm

# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
# model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# optimizer = AdamW(model.parameters(), lr=5e-5)

class BiLSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, dropout_rate, out_dim, batch_size, device):
		super(BiLSTM, self).__init__()
		self.batch_size = batch_size
		self.embed_dim = embedding_dim
		self.hidden_size = hidden_dim
		self.directions = 2
		self.num_layers = 2
		self.concat = 4
		self.projection = nn.Linear(self.embed_dim, self.hidden_size)
		self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers,
									bidirectional = True, batch_first = True, dropout = dropout_rate)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p = dropout_rate)

		self.lin1 = nn.Linear(self.hidden_size * self.directions * self.concat, self.hidden_size)
		self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.lin3 = nn.Linear(self.hidden_size, out_dim)

		self.device = device

		for lin in [self.lin1, self.lin2, self.lin3]:
			nn.init.xavier_uniform_(lin.weight)
			nn.init.zeros_(lin.bias)

		self.out = nn.Sequential(
			self.lin1,
			self.relu,
			self.dropout,
			self.lin2,
			self.relu,
			self.dropout,
			self.lin3
		)

	def forward(self, batch):
		premise_embed = batch[:, 0:1, :]
		hypothesis_embed = batch[:, 1:2, :]

		premise_proj = self.relu(self.projection(premise_embed))
		hypothesis_proj = self.relu(self.projection(hypothesis_embed))

		h0 = c0 = torch.tensor([]).new_zeros((self.num_layers * self.directions, batch.shape[0], self.hidden_size)).to(self.device)

		_, (premise_ht, _) = self.lstm(premise_proj, (h0, c0))
		_, (hypothesis_ht, _) = self.lstm(hypothesis_proj, (h0, c0))
		
		premise = premise_ht[-2:].transpose(0, 1).contiguous().view(batch.shape[0], -1)
		hypothesis = hypothesis_ht[-2:].transpose(0, 1).contiguous().view(batch.shape[0], -1)

		combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), 1)
		return self.out(combined)