import os
import torch
import transformers
from minivnGPT.model import minivnGPT, ModelArguments
from train import TrainingArguments

training_args = TrainingArguments()

model_args_dict = {}
ckpt_path = os.path.join(training_args.out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=training_args.device)
checkpoint_model_args = checkpoint["model_args"]
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ["n_layer", "n_head", "d_model", "context_length", "bias", "vocab_size"]:
    model_args_dict[k] = getattr(checkpoint_model_args, k)
# create the model
model_args = ModelArguments(**model_args_dict)

# Tokenize
tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=training_args.tokenizer_path,
    model_max_length=model_args.context_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
model_args.vocab_size = tokenizer.vocab_size

model = minivnGPT(model_args)
state_dict = checkpoint["model"]
# fix the keys of the state dictionary
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
# Load state dict to model
model.load_state_dict(state_dict)

model.eval()

input = "Cô ấy là một phụ nữ đẹp"
enc_input = tokenizer(
    input,
).input_ids

enc_input = torch.tensor(enc_input).type(torch.long).view(1, -1)
print(f"input tensor shape: {enc_input.shape}")

output = tokenizer.decode(
    model.generate(enc_input, 500)[0].tolist(), skip_special_tokens=True
)
print(output)
