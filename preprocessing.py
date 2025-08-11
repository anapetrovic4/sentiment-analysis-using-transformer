import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import os


"""
Load CSV as Pandas DataFrame
"""
df = pd.read_csv('data/data.csv')
print(df.head())
print("\n")

"""
Train and test split 
"""
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Sentiment']) # Stratify is used since classes are imbalanced
print("Train set")
print(train.head())
print("\n")
print("Test set")
print(test.head())
print("\n")

# CHECK DISTRIBUTION
distribution = train['Sentiment'].value_counts(normalize=True)
print(distribution)


"""
Case folding: Optional
"""
train['Sentence'] = train['Sentence'].str.lower()
print("Train set case folded")
print(train.head())
print("\n")
test['Sentence'] = test['Sentence'].str.lower()
print("Test set case folded")
print(test.head())
print("\n")

"""
Tokenization
"""
train['Sentence'] = train['Sentence'].str.split()
print("Train set tokenized")
print(train.head())
print("\n")
test['Sentence'] = test['Sentence'].str.split()
print("Test set tokenized")
print(test.head())
print("\n")

"""
Create vocabulary from train set
"""
train_tokenized = train['Sentence'].tolist() # Convert all tokens to a list
print(train_tokenized[:5]) 

# Count all tokens
counter = Counter()
for tokens in train_tokenized:
    counter.update(tokens)

# Create vocab with min_freq and special tokens
min_freq = 2 # Remove any token that appears less than 2 times
vocab = {"<PAD>" : 0, "<UNK>" : 1} # <PAD> for padding, <UNK> for unknown (rare words)
for token, freq in counter.items():
    if freq >= min_freq:
        vocab[token] = len(vocab)

# Check 20 first tokens
print("Check first 20 tokens")
for token, idx in list(vocab.items())[:20]: # Checking to see if tokenization is performed well, or if there are unusual words that are occuring
   print(f"{token}: {idx}")

# Check vocabulary size
print(f"Vocab size: {len(vocab)}")

# Check most common tokens
print("Most common tokens")
for word, freq in counter.most_common(20):
    print(f"{word}: {freq}")

# Check unusual tokens: Improvement: Add <NUM> because there are a lot of numbers
for token in vocab:
    if token.isnumeric() or not token.isalnum():
        print(token)

"""
Map tokens to IDs
"""
train['input_ids'] = train['Sentence'].apply(
    lambda tokens: [vocab.get(token, vocab["<UNK>"]) for token in tokens]
)
print("Train set with mapped tokens")
print(train.head())
print("\n")
test['input_ids'] = test['Sentence'].apply(
    lambda tokens: [vocab.get(token, vocab["<UNK>"]) for token in tokens]
)
print("Test set with mapped tokens")
print(test.head())
print("\n")

"""
Sentence length analysis
1. Count number of tokens in each row
2. Get a list of number of tokens in each row
3. Calculate:
min length
max length
average length
median
"""

# 1. Count number of Sentence in each row
token_lengths = train['Sentence'].apply(len)

# 2. Get a list of number of Sentence in each row
length_list = token_lengths.tolist()

# 3. Calculate statistics
min_len = np.min(length_list)
max_len = np.max(length_list)
avg_len = np.mean(length_list)
median_len = np.median(length_list)

print(f"Min length: {min_len}")
print(f"Max length: {max_len}")
print(f"Average length: {avg_len:.2f}")
print(f"Median length: {median_len}")

"""
Add padding and attention masks
"""
def pad_sequence(seq, max_len, pad_id):
    seq = seq[:max_len]  # Cut if it's too long
    padding = [pad_id] * (max_len - len(seq)) # Calculate how many PAD tokens are needed
    return seq + padding 

def create_attention_mask(seq, max_len):
    seq = seq[:max_len] # Cut if it's too long
    return [1]*len(seq) + [0]*(max_len - len(seq)) # Add 1 where the token is present, add 0 for padding

MAX_LEN = 64 # Based on Sentence length analysis: 64
PAD_ID = vocab["<PAD>"]

train['padded_input_ids'] = train['input_ids'].apply(
    lambda x: pad_sequence(x, MAX_LEN, PAD_ID)
)

train['attention_mask'] = train['input_ids'].apply(
    lambda x: create_attention_mask(x, MAX_LEN)
)

# Same for the test set
test['padded_input_ids'] = test['input_ids'].apply(
    lambda x: pad_sequence(x, MAX_LEN, PAD_ID)
)

test['attention_mask'] = test['input_ids'].apply(
    lambda x: create_attention_mask(x, MAX_LEN)
)

"""
Label encoding
"""
label2id = {"positive": 0, "neutral": 1, "negative": 2}
train['label'] = train['Sentiment'].map(label2id)
test['label'] = test['Sentiment'].map(label2id)

print(train.iloc[0])
print(test.iloc[0])

print(len(train.iloc[0]['padded_input_ids'])) 
print(len(train.iloc[0]['attention_mask']))    

train['padded_input_ids'].apply(len).value_counts()

print("Input IDs:", train.iloc[0]['input_ids']) 
print("Paddded nput IDs:", train.iloc[0]['padded_input_ids']) # Sentences shorter than 64 tokens have <PAD> (0) until length=64 is satisfied
print("Mask     :", train.iloc[0]['attention_mask']) # Real token: 1, Padding: 0 

print("Sentiment label:", train.iloc[0]['Sentiment'])
print("Label ID:", train.iloc[0]['label'])

"""
Add word cloud for each class
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

positive_text = " ".join(train[train['Sentiment'] == "positive"]['Sentence'].apply(lambda x: " ".join(x)))
neutral_text = " ".join(train[train['Sentiment'] == "neutral"]['Sentence'].apply(lambda x: " ".join(x)))
negative_text = " ".join(train[train['Sentiment'] == "negative"]['Sentence'].apply(lambda x: " ".join(x)))

def plot_wordcloud(text, filename, title):
    wc = WordCloud(width=600, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

plot_wordcloud(positive_text, "imgs/positive_wordcloud.png", "Positive Sentiment Word Cloud")
plot_wordcloud(neutral_text, "imgs/neutral_wordcloud.png", "Neutral Sentiment Word Cloud")
plot_wordcloud(negative_text, "imgs/negative_wordcloud.png", "Negative Sentiment Word Cloud")

# Save processed data
train.to_pickle("data/processed_train.pkl")
test.to_pickle("data/processed_test.pkl")

import pickle
from config import PAD_TOKEN

# Vocab
if not os.path.exists("data/vocab.pkl"):
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

# PAD ID
if not os.path.exists("data/pad_id.pkl"):
    with open("data/pad_id.pkl", "wb") as f:
        pickle.dump(vocab[PAD_TOKEN], f)