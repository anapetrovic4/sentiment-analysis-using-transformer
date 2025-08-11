import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import FinancialDataset
import pickle
from transformer import TransformerClassifier
from config import MAX_LEN
from evaluate import evaluate_model
from train import train_model

# Load vocab
with open("data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load preprocessed data
train_df = pd.read_pickle("data/processed_train.pkl")
test_df = pd.read_pickle("data/processed_test.pkl")

# Initialize Dataset
train_dataset = FinancialDataset(train_df)
test_dataset = FinancialDataset(test_df)

# Initialize DataLoader

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Check batch dimensions

batch = next(iter(train_loader))
print("input_ids shape:", batch['input_ids'].shape)         # torch.Size([32, 64])
print("attention_mask shape:", batch['attention_mask'].shape)
print("label shape:", batch['label'].shape) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerClassifier(vocab_size=len(vocab), max_len=MAX_LEN).to(device)

train_model(model, train_loader, device)
evaluate_model(model, test_loader, device)
