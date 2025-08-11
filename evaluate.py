import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import ID2LABEL
from datetime import datetime
import os

os.makedirs("imgs", exist_ok=True) # Create folder to store metric imaages

def evaluate_model(model, test_loader, device):
    model.eval() # Put model in evaluation mode: turn off dropout
    correct = 0
    total = 0

    true_labels = []
    pred_labels = []

    with torch.no_grad(): # Turn off gradient tracking to make the computatiaon faster
        for batch in test_loader: # Load data from test set
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=1) # Calculate prediction with the highest accuracy from logits
            correct += (preds == labels).sum().item() # Update number of accurate predictions 
            total += labels.size(0) # Update number of total number of predictions

            true_labels.extend(labels.cpu().numpy()) # Store true labels 
            pred_labels.extend(preds.cpu().numpy()) # Store predicted labels

    print(f"Test Accuracy: {correct/total:.4f}") # Calculate total accuracy
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())] # 
    print(classification_report(true_labels, pred_labels, target_names=target_names)) # Calculate precision, recall, F1-score per class

    # Calculate macro averages - useful for imbalanced classes
    print(f"Overall Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
    print(f"Macro Precision:  {precision_score(true_labels, pred_labels, average='macro'):.4f}")
    print(f"Macro Recall:     {recall_score(true_labels, pred_labels, average='macro'):.4f}")
    print(f"Macro F1:         {f1_score(true_labels, pred_labels, average='macro'):.4f}")

    # Generate confusion matrix plot 
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/confusion_matrix_{timestamp}.png")
    plt.close()

    # Generate F1-score per class 
    f1_scores = f1_score(true_labels, pred_labels, average=None)
    sns.barplot(x=target_names, y=f1_scores)
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title("F1-score per class")
    plt.savefig("imgs/f1_per_class.png")
    plt.close()
