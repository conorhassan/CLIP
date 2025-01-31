from datasets import load_dataset
import torchvision.transforms as T
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def peek_at_batches(batch_size=5):
    # Load dataset
    dataset = load_dataset("eltorio/ROCOv2-radiology")  # replace with actual ROCO dataset
    
    # Basic transforms
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
    
    # Get a batch
    batch = dataset['train'].select(range(batch_size))
    
    # Process and visualize
    fig, axes = plt.subplots(1, batch_size, figsize=(20, 4))
    
    for i, item in enumerate(batch):
        # Process image
        img = transforms(item['image'].convert('RGB'))
        # Denormalize for visualization
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        
        # Process text
        tokens = tokenizer(item['caption'], truncation=True, max_length=77)
        
        # Plot
        axes[i].imshow(img.permute(1,2,0).clip(0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"Tokens: {len(tokens['input_ids'])}", fontsize=8)
        print(f"Caption {i}: {item['caption']}\n")
    
    plt.tight_layout()
    plt.show()

# Use it
peek_at_batches(5)