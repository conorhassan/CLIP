import torch
from transformers import CLIPModel as HFCLIPModel, CLIPTokenizer, CLIPProcessor
from src.model import CLIPModel, CLIPConfig

def load_hf_weights(our_model, hf_model):
    # Load state dicts from the Hugging Face model for text and vision branches
    hf_text_state_dict = hf_model.text_model.state_dict()
    hf_vision_state_dict = hf_model.vision_model.state_dict()

    print("Loading text model weights...")
    missing_text, unexpected_text = our_model.text_model.load_state_dict(hf_text_state_dict, strict=False)
    print("Text model missing keys:", missing_text)
    print("Text model unexpected keys:", unexpected_text)

    print("Loading vision model weights...")
    missing_vision, unexpected_vision = our_model.vision_model.load_state_dict(hf_vision_state_dict, strict=False)
    print("Vision model missing keys:", missing_vision)
    print("Vision model unexpected keys:", unexpected_vision)

    return our_model

def compare_outputs(our_model, hf_model, tokenizer, text, image):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        # Get outputs from our model
        our_text_out, our_image_out = our_model(tokens.input_ids, image, tokens.attention_mask)
        # Get outputs from the HF model; note that the HF model returns a CLIPOutput object
        hf_outputs = hf_model(input_ids=tokens.input_ids, pixel_values=image)
        hf_text_out = hf_outputs.text_embeds  # use text_embeds instead of text_projection
        hf_image_out = hf_outputs.image_embeds  # use image_embeds instead of vision_projection
    
    text_diff = torch.max(torch.abs(our_text_out - hf_text_out)).item()
    image_diff = torch.max(torch.abs(our_image_out - hf_image_out)).item()
    print("Difference in text outputs:", text_diff)
    print("Difference in image outputs:", image_diff)
if __name__ == '__main__':
    # Initialize tokenizer and processor from Hugging Face
    hf_model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(hf_model_name)
    processor = CLIPProcessor.from_pretrained(hf_model_name)
    
    # Create our CLIP model with default config
    clip_config = CLIPConfig()
    our_model = CLIPModel(clip_config)
    
    # Load Hugging Face CLIP model for comparison
    hf_model = HFCLIPModel.from_pretrained(hf_model_name)
    
    # Load HF weights into our model
    our_model = load_hf_weights(our_model, hf_model)
    
    # Prepare sample inputs: sample text and dummy image tensor
    sample_text = "A photo of a unicorn in a garden"
    # Use the processor to get a properly preprocessed image; here we use a random tensor for demonstration
    # For a real image, you might use PIL.Image.open or similar.
    dummy_image = torch.randn(1, 3, 224, 224)

    # Compare outputs on sample inputs
    compare_outputs(our_model, hf_model, tokenizer, sample_text, dummy_image)