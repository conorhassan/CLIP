from datasets import load_dataset
import torchvision.transforms as T
from transformers import AutoTokenizer
import matplotlib.pyplot as plt 
import numpy as np

# # Load just a tiny slice - this will download minimal data
# dataset = load_dataset("eltorio/ROCOv2-radiology", split="train[:10]")  # only get first 10 examples

# # Try this instead:xs
# dataset = load_dataset("eltorio/ROCOv2-radiology", split="train", streaming=True)
# dataset = dataset.take(10)  # Get first 10 examples

dataset = load_dataset("ylecun/mnist", split="train[:10]")

# Setup minimal transforms
transforms = T.Compose([
    T.Resize((128, 128)), # (224, 224)
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")

# Let's look at what we got
print(f"Dataset size: {len(dataset)}")
print("\nDataset features:", dataset.features)

# Look at first example raw data
print("\nFirst example:")
print(dataset[0])

# Show transformed version
example = dataset[0]

transformed_img = transforms(example['image'].convert('RGB'))

# Plot original and transformed image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# Original image
ax1.imshow(example['image'], cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

# Transformed image
transformed_np = transformed_img.numpy()
transformed_np = np.transpose(transformed_np, (1,2,0))  # Change from CxHxW to HxWxC
ax2.imshow(transformed_np)
ax2.set_title('Transformed Image (16x16)')
ax2.axis('off')

plt.show()

# # NOTE: for when we are using the ROCCO dataset 
# need to check the RICO dataset tbh
# tokens = tokenizer(example['caption'], truncation=True, max_length=77)

# print("\nTransformed shapes:")
# print(f"Image tensor shape: {transformed_img.shape}")
# print(f"Token length: {len(tokens['input_ids'])}")