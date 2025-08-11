import torch
import numpy as np
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from config import LABEL2ID

def train_model(model, train_loader, device):
    labels = [x['label'].item() for x in train_loader.dataset] # Weights are prepared for imbalanced classes. Item() converts tensors to scalars
    class_weights = compute_class_weight( 
        class_weight='balanced', # We want to balance classes by adding a bigger weight to the sparse class
        classes=np.array(list(LABEL2ID.values())),
        y=np.array(labels)
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # CrossEntropyLoss penalizes error on sparse classes stricter 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Adam optimizer with a small learning rate is used

    EPOCHS = 50
    for epoch in range(EPOCHS):
        model.train() # Train in batches
        total_loss = 0
        correct = 0 
        total = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad() # Reset gradients 
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels) # Output type: Logits from each sample in a batch
            loss.backward()
            optimizer.step() # Calculate loss and gradients for every model parameter

            total_loss += loss.item() # Calculate loss for one epoch
            preds = outputs.argmax(dim=1) # Get prediction with the highest score
            correct += (preds == labels).sum().item() # Calculate correctly classified samples from a batch
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - Accuracy: {correct/total:.4f}")
