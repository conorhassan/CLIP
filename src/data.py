import torch 
from torchvision import transforms
from torch.utils.data import IterableDataset, DataLoader

from transformers import CLIPTokenizer
from datasets import load_dataset

from PIL import Image

class CLIPImageProcessor: 
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __call__(self, image):
        """
        Args: 
            image: PIL image or a list of PIL images
        Returns: 
            preprocessed image tensor
        """
        if isinstance(image, list):
            return torch.stack([self.transform(img) for img in image])
        else:
            return self.transform(image)
        
class ClipStreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __iter__(self):
            for item in self.dataset:
                image = item["image"]
                if not isinstance(image, Image.Image): 
                    image = Image.fromarray(image)

                image_tensor = self.image_processor(image.convert("RGB"))

                caption = item["caption"]
                text_tokens = self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                    )
                    
                yield {
                    'pixel_values': image_tensor,
                    'input_ids': text_tokens['input_ids'].squeeze(0), 
                    "attention_mask": text_tokens['attention_mask'].squeeze(0)
                }

def create_streaming_clip_loader(
    dataset="eltorio/ROCOv2-radiology", 
    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32"), 
    image_processor=CLIPImageProcessor(), 
    batch_size=128, 
    num_workers=1, 
    pin_memory=True, 
    shuffle=False, 
    subset=1024
):
    dataset = load_dataset(dataset, split="train", streaming=True)
    if subset != None: 
        dataset = dataset.take(subset)
    dataset = ClipStreamingDataset(dataset, tokenizer, image_processor)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    
if __name__ == "__main__":
    loader = create_streaming_clip_loader()
    for batch in loader:
        print(batch)
        break