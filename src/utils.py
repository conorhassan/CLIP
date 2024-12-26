import torch
from einops import repeat

def tokenize_and_mask(texts, tokenizer):
    outputs = tokenizer(texts, padding=True, return_tensors="pt")
    mask = repeat(outputs.attention_mask, "b s -> b 1 s t", t=outputs.attention_mask.shape[1])
    mask = (1.0 - mask) * torch.finfo(torch.float32).min
    return outputs.input_ids, mask

# # EXAMPLE USAGE:
# from transformers import CLIPTokenizer

# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# # Example with padding:
# texts = [
#     "A very short text",
#     "A much longer text that will need more tokens and show padding in action"
# ]

# outputs = tokenizer(
#     texts,
#     padding=True,
#     return_tensors="pt"
# )

# print("Input IDs shape:", outputs.input_ids.shape)
# print("Attention mask:", outputs.attention_mask)
