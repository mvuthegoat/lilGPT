import os
import numpy as np
import transformers, datasets


def preprocess(
    tokenizer: transformers.PreTrainedTokenizer,
):
    dataset = datasets.load_dataset("wiki_lingua", "vietnamese")  # wikihow texts
    data = [d.get("document")[0] for d in dataset["train"]["article"]]
    # Add a EOS token to the end of each example
    data = [s + tokenizer.eos_token for s in data]
    # Tokenize texts
    input_ids = tokenizer(
        data,
    ).input_ids

    # Split train/test. Default split is 0.90
    np.random.seed(0)
    perm = np.random.permutation(len(input_ids))
    split = int(len(perm) * 0.95)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_data = [input_ids[i] for i in train_indices]
    eval_data = [input_ids[i] for i in eval_indices]

    # Concatenate examples to make a single list of input
    train_data = [item for example in train_data for item in example]
    eval_data = [item for example in eval_data for item in example]

    train_data = np.array(train_data)
    eval_data = np.array(eval_data)

    return train_data, eval_data
