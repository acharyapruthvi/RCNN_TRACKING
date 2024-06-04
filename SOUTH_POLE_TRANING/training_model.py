import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Dataset loading function
def load_data(base_raw_dir, base_outlined_dir, my_years, img_size=(1156, 1156), test_size=0.2):
    images = []
    masks = []
    for year in my_years:
        raw_dir = os.path.join(base_raw_dir, f"MY_{year}")
        outlined_dir = os.path.join(base_outlined_dir, f"MY_{year}")
        for filename in os.listdir(raw_dir):
            print(filename)
            img_path = os.path.join(raw_dir, filename)
            mask_path = os.path.join(outlined_dir, filename)
            if os.path.exists(mask_path):
                img = imread(img_path)
                mask = imread(mask_path)
                img = resize(img, img_size, preserve_range=True)
                mask = resize(mask, img_size, preserve_range=True)
                images.append(img / 255.0)
                masks.append(mask[:,:,1] > 128)
    images = np.array(images)
    masks = np.array(masks)
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=test_size, random_state=42)
    return X_train, y_train, X_val, y_val

# Custom Dataset Class for Segmentation
# Adjustments to DataLoader to ensure the tensors are correctly formatted
class MarsCapSegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)  # Convert HWC to CHW
        mask = torch.tensor(self.masks[idx], dtype=torch.long)  # Ensure mask is in long format
        return img, mask

# Model setup for Semantic Segmentation
def get_segmentation_model(num_classes):
    model = deeplabv3_resnet50(pretrained=True)
    # Change the output classifier to match the number of classes
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def train_model_segmentation(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Training')
    for batch_idx, (images, masks) in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = torch.nn.functional.cross_entropy(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

    return total_loss / len(data_loader)

def validate_model_segmentation(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = torch.nn.functional.cross_entropy(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Main Execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_raw_dir = '/home/pruthvi/Desktop/MARCI_VIDEOS/South_Pole'
    base_outlined_dir = '/home/pruthvi/Desktop/MARCI_VIDEOS/CODE/SOUTH_POLE/OUTLINES'
    my_years = [29, 30]

    # Load training and validation data
    train_images, train_masks, val_images, val_masks = load_data(base_raw_dir, base_outlined_dir, my_years)
    train_dataset = MarsCapSegmentationDataset(train_images, train_masks)
    val_dataset = MarsCapSegmentationDataset(val_images, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = get_segmentation_model(num_classes=2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    print("Starting Training")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_model_segmentation(model, train_loader, optimizer, device)
        val_loss = validate_model_segmentation(model, val_loader, device)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    model_save_path = '/home/pruthvi/Desktop/MARCI_VIDEOS/CODE/NORTH_POLE/NP_IS.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
