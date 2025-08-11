MAX_LEN = 64 # Defining optimal token length, observed from token analysis 
PAD_TOKEN = "<PAD>" # Defining padding token
UNK_TOKEN = "<UNK>" # Defining unknown token 
MIN_FREQ = 2 # Defining minimum frequency; if any word occurences less than min_freq, it will be ignored 
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2} # Label encoding of "Sentiment" target attribute 
ID2LABEL = {v: k for k, v in LABEL2ID.items()} # Reverse process of label encoding

