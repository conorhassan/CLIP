from src.loss import ConstrastiveLoss
import torch
from torch import optim
from src.model import CLIPModel, CLIPConfig
from src.data import create_streaming_clip_loader

def train_clip(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    criterion = ConstrastiveLoss()
    
    for batch in dataloader:
        batch_count += 1
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        text_features, image_features = model(input_ids, pixel_values, attention_mask)
        loss = criterion(text_features, image_features)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    return avg_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize your CLIP model and move to device
    clip_config = CLIPConfig()
    model = CLIPModel(clip_config).to(device)
    
    # Create data loader for the ROCO dataset
    dataloader = create_streaming_clip_loader(batch_size=8)
    
    print("Finished the creation of the data stream.")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        avg_loss = train_clip(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
if __name__ == "__main__":
    main()