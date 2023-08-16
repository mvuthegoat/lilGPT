import os
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer

## dataset used: "Libosa2707/vietnamese-poem" from huggingface dataset

dataset = load_dataset("Libosa2707/vietnamese-poem")
#print(dataset)

dataset = dataset["train"].train_test_split(test_size=0.1)
#dataset['validation'] = dataset['test']  # change the dict key test to valid
#del dataset['test']

## sample some dataset so that we can reduce training time.
dataset["train"] = dataset["train"].select([i for i in range(8000)])
dataset["validation"] = dataset["validation"].select([i for i in range(2000)])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
#print(tokenizer.is_fast)
#print(tokenizer.vocab_size)
train_data = tokenizer.encode(str(dataset["train"]["content"]))
val_data = tokenizer.encode(str(dataset["validation"]["content"]))
``
train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")
print(f"The id of the word yêu in the tokenizer: {tokenizer.encode('yêu')}") # output: The id of the word yêu in the tokenizer: [101, 30162, 102]
