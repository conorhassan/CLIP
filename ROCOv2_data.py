from datasets import load_dataset
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms import ToTensor

dataset = load_dataset("eltorio/ROCOv2-radiology", split="train", streaming=True)

ds = dataset.take(256)

item = next(iter(ds))

print("Caption: ", item['caption'])

# Create transforms
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to standard size
    ToTensor(),          # Convert to tensor
    # Add any other transforms you want, e.g.:
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Get your PIL image
pil_image = item['image']  # Your PIL image

# Create transformed version
tensor_image = transform(pil_image)

# Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot original
ax1.imshow(pil_image)
ax1.set_title('Original')
ax1.axis('off')

# Plot transformed (need to denormalize and convert back if normalized)
if tensor_image.shape[0] == 3:  # If image is in CHW format
    # Denormalize if you used normalization
    denorm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    display_tensor = denorm(tensor_image)
    
    # Convert to PIL for display
    display_image = T.ToPILImage()(display_tensor)
    ax2.imshow(display_image)
    ax2.set_title('Transformed')
    ax2.axis('off')

plt.show()
