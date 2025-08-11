import torch.nn as nn
import torch
import numpy as np
import pickle

with open("data/pad_id.pkl", "rb") as f:
    PAD_ID = pickle.load(f)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # Initialize zero-like matrix with dims (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Initialize vector with positions (0, 1, 2, ... , max_len-1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)) # Frequency of sin/cos functions depends on position and d_model
        pe[:, 0::2] = torch.sin(position * div_term) # Even indices: sinus
        pe[:, 1::2] = torch.cos(position * div_term) # Odd indices: cosinus
        pe = pe.unsqueeze(0) # (1, max_len, d_model) # Adds new dimension for the batch
        self.register_buffer('pe', pe) # Signals that 'pe' is not a parameter used for training, but it should be saved in the model

    
    def forward(self, x): 
        x = x + self.pe[:, :x.size(1), :] # Adds positional encoding to input vectors by sequence length
        return x
    

class TransformerClassifier(nn.Module):
    def __init__(self,
                vocab_size, # Number of tokens in vocabulary
                embed_dim=128, # Dimensions of embedded vector
                num_heads=4, # Multi-head attention
                hidden_dim=256, # The width of feedforward layer inside of encoder
                num_layers=2, # Number of layers in encoder
                num_classes=3, # Number of values in target variable
                max_len=64, # Max length of a sequence
                droput=0.3): # Dropout possibility
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID) # Word embedding layer: learns vectors for each token in vocabulary. Padding token is ignored during training
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len) # Positional encoding is done through embedding

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=droput) # One layer is defined; stack is formed after
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.dropout = nn.Dropout(droput) # Regularization technique: Dropout
        self.fc = nn.Linear(embed_dim, num_classes) # Linear layer for classification to num_classes

    def forward(self, input_ids, attention_mask):
        
        embedded = self.embedding(input_ids) # Add embedding
        embedded = self.pos_encoder(embedded) # Add positional encoding

        embedded = embedded.permute(1, 0, 2) # Expects: (batch_size, seq_len, embed_dim)

        src_key_padding_mask = attention_mask == 0 # Mask attention where padding is 0 
        
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask) 
        transformer_output = transformer_output.permute(1, 0, 2)

        cls_output = transformer_output.mean(dim=1) # Mean pooling

        output = self.dropout(cls_output) # Dropout appliance
        logits = self.fc(output) # Linear classification
        return logits # Logits are values before softmax values for each class