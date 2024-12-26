# import torch
# from text_model import TextModel, TextModelConfig
# from transformers import CLIPModel, CLIPTokenizer

# # Create config and model
# config = TextModelConfig()
# model = TextModel(config)

# # Create sample input
# batch_size = 2
# seq_len = 5  # Should be <= max_position_dim (77)
# x = torch.randint(0, config.vocab_dim, (batch_size, seq_len))  # Random token ids

# # Run forward pass
# output = model(x)

# # Print shapes at each step
# print(f"Input shape: {x.shape}")  # Should be [2, 5]
# print(f"Output shape: {output.shape}")  # Should be [2, 5, 512]

# # Optional: Check parameter count
# total_params = sum(p.numel() for p in model.parameters())
# print(f"\nTotal parameters: {total_params:,}")

# # Verify device and dtype
# print(f"\nInput device: {x.device}")
# print(f"Output device: {output.device}")
# print(f"Output dtype: {output.dtype}")


# # LOAD HUGGING FACE MODEL WEIGHTS INTO OUR MODEL! 

# def load_hf_weights(our_model, pretrained_name="openai/clip-vit-base-patch32"):
#     # Load full CLIP model
#     hf_model = CLIPModel.from_pretrained(pretrained_name)
#     hf_state_dict = hf_model.text_model.state_dict()  # Get just text model weights
    
#     # Remove 'text_model.' prefix from keys
#     new_state_dict = {}
#     for key, value in hf_state_dict.items():
#         new_key = key  # Keep original key structure
#         new_state_dict[new_key] = value
    
#     # Load weights
#     missing, unexpected = our_model.load_state_dict(new_state_dict, strict=False)
#     print(f"Missing keys: {missing}")
#     print(f"Unexpected keys: {unexpected}")
    
#     return our_model

# def compare_outputs(our_model, text="A photo of a cat"):
#     hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
#     tokens = tokenizer(text, return_tensors="pt", padding=True)
#     input_ids = tokens.input_ids
    
#     with torch.no_grad():
#         # Check embeddings first
#         our_embeds = our_model.embeddings(input_ids)
#         hf_embeds = hf_model.text_model.embeddings(input_ids)
#         print(f"Embedding diff: {torch.max(torch.abs(our_embeds - hf_embeds))}")
        
#         # Check first layer output
#         our_first_layer = our_model.encoder.layers[0](our_embeds)
#         hf_first_layer = hf_model.text_model.encoder.layers[0](hf_embeds)
#         print(f"First layer diff: {torch.max(torch.abs(our_first_layer - hf_first_layer))}")
        
#         # Full outputs
#         our_output = our_model(input_ids)
#         hf_output = hf_model.text_model(input_ids)[0]
#         print(f"Final diff: {torch.max(torch.abs(our_output - hf_output))}")

import torch
from transformers import CLIPTokenizer
from text_model import TextModel, TextModelConfig

if __name__ == '__main__':
    # config = TextModelConfig()
    # model = TextModel(config)
    # model = load_hf_weights(model)

    # compare_outputs(model)

    # # EXAMPLE USAGE:

# Initialize tokenizer and model
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    config = TextModelConfig()
    model = TextModel(config)

    # Sample text
    text = "A photo of a unicorn in a garden"

    # Tokenize
    tokens = tokenizer(
        text, 
        padding=True, 
        return_tensors="pt"
    )

    # Forward pass
    with torch.no_grad():
        output = model(
            tokens.input_ids,
            tokens.attention_mask
        )

    print(f"Input shape: {tokens.input_ids.shape}")
    print(f"Mask shape: {tokens.attention_mask.shape}")
    print(f"Output shape: {output.shape}")
    print(output[0, :, :10])
