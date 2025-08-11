import torch
from torch.utils.data import Dataset

class FinancialDataset(Dataset):
    def __init__(self, df):
        self.input_ids = df['padded_input_ids'].tolist() # Lists of padded tokens until MAX_LEN
        self.attention_masks = df['attention_mask'].tolist() # Masks that show if tokens are real (1) or masks (0)
        self.labels = df['label'].tolist() # Numerical IDs for sentiment 

    def __len__(self):
        return len(self.labels) # Get the length of a sample 
    
    def __getitem__(self, idx): # Load a sample by index in a dictionary form
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long), # Input ID tokens; PyTorch expects type Long for NLP models
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long), # Attention mask
            'label': torch.tensor(self.labels[idx], dtype=torch.long) # Label 
        }
    
    