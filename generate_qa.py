import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    functional as F,
)  # can be directly applied to input tensors, don't need to create a module to use it
from transformers import AutoTokenizer
from minivnGPT.model import minivnGPT, minivnGPTConfig, minivnGPTForQA

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

# Initialize models
model_args = minivnGPTConfig()
model = minivnGPT(model_args)
pre_trained_ckpt = torch.load(
    "/home/minhvn/workspace/llm/my_model/checkpoints/ckpt-2000.pt"
)
model.load_state_dict(pre_trained_ckpt["model"])
qa_model = minivnGPTForQA(model, model_args)
fine_tuned_ckpt = torch.load(
    "/home/minhvn/workspace/llm/my_model/checkpoints/ckpt-qa-200.pt"
)
qa_model.load_state_dict(fine_tuned_ckpt["model"])

"""
question_test = "Thành phố Miaoli nằm ở quốc gia nào??"
context_test = "Thành phố Miêu Lật tiếng Trung:苗栗市, Bính âm:Miáolì Shì, Pe̍h-ōe-jī:Biâu-le̍k-chhī) là huyện lỵ của Huyện Miêu Lật, Đài Loan. Từ Miêu Lậtlà kết hợp của hai từ trong tiếng Khách Gia là mèo (貓) và thành phố (裡), được phát âm gần như là Pali (Bari) trong các ngôn ngữ của Thổ dân Đài Loan. Thành phố có tỷ lệ người Khách Gia cao nhất tại Đài Loan. Năm 2009, dân số thành phố là 90.209 người trên tổng diện tích là 37,8878 km²"
"""
question_test = "hôm nay bạn khoẻ không. người giàu nhất thế giới là ai?"
context_test = "Thành phố Miêu Lật tiếng Trung:苗栗市, Bính âm:Miáolì Shì, Pe̍h-ōe-jī:Biâu-le̍k-chhī) là huyện lỵ của Huyện Miêu Lật, Đài Loan. Từ Miêu Lậtlà kết hợp của hai từ trong tiếng Khách Gia là mèo (貓) và thành phố (裡), được phát âm gần như là Pali (Bari) trong các ngôn ngữ của Thổ dân Đài Loan. Thành phố có tỷ lệ người Khách Gia cao nhất tại Đài Loan. Năm 2009, dân số thành phố là 90.209 người trên tổng diện tích là 37,8878 km²"
# context_test = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
tokenized_examples = tokenizer(
    question_test,
    context_test,
    padding="max_length",
    max_length=256,  # same as model context_length
    truncation="only_second",
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(type(tokenized_examples))
print(tokenized_examples.keys())
# print(tokenized_examples['input_ids'])
# print(len(tokenized_examples['input_ids']))
input_ids = torch.tensor(tokenized_examples["input_ids"]).type(torch.long)
print((input_ids))
# print(input_ids)

start, end = qa_model.generate(input_ids)
print(start)
print(end)

print("\n")
print("Question: ")
print(question_test)
print(" ")
print("Context: ")
print(context_test)
print(" ")
print("Answer: ")
if start > end:
    print("THERE IS NO ANSWER FOR THIS QUESTION. TRY AGAIN")
else:
    print("\n")
    enc_context = tokenizer.encode(context_test)

    output_sentence = enc_context[start : end + 1]
    print(output_sentence)
    print(tokenizer.decode(output_sentence))

"""
print(' ')
print('Loss: ')
print(loss)
"""
