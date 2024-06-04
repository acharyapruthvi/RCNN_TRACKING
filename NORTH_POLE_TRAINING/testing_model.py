import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import torch
import cv2
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP

def load_model(model_path, num_classes):
    # Initialize the model with the auxiliary classifier as used in default pretrained models
    # The default auxiliary classifier uses a more complex structure, adjust these as per your training setup
    # This code assumes that the default structure was used; adjust if different
    model = deeplabv3_resnet50(pretrained=False, aux_loss=True)

    model.classifier = DeepLabHead(2048, num_classes)  # Change the classifier
    
    # Load the state dictionary
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()
    return model
# Function to prepare an image in the same format as used during training
def prepare_image(image_path):
    img = imread(image_path)
    img_resized = resize(img, (1156, 1156), preserve_range=True) # Redundant step to ensure the image is of the same size as used during training
    img_tensor = torch.tensor(img_resized, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # Normalize the image to 0-1
    img_batch = img_tensor.unsqueeze(0)
    return img_batch

# Function that predicts the class of an image
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
        return output.argmax(0)

# Function to load mask image for comparison
def load_actual_mask(mask_path):
    mask = imread(mask_path)
    mask = resize(mask, (1156, 1156), preserve_range=True)
    binary_mask = (mask[:,:,1] > 128).astype(np.uint8)  # Assuming green channel represents the mask
    return binary_mask

# Function to calculate Intersection over Union (IoU)
def calculate_iou(predicted_mask, actual_mask):
    intersection = np.logical_and(predicted_mask, actual_mask)
    union = np.logical_or(predicted_mask, actual_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Load model and predict
model_path = '/home/pruthvi/Desktop/MARCI_VIDEOS/CODE/NORTH_POLE/NP_IS.pth'
model = load_model(model_path, num_classes=2)
image_path = '/home/pruthvi/Desktop/MARCI_VIDEOS/North_Pole/TRACKING/MY_Frames/MY_35/14.6.png'
image_png = cv2.imread(image_path)
image_tensor = prepare_image(image_path)
predicted_mask = predict(model, image_tensor).numpy() == 1  # Class 1 for the cap

# Load actual mask and calculate IoU
actual_mask_path = '/home/pruthvi/Desktop/MARCI_VIDEOS/North_Pole/TRACKING/GOOD_TRACKING/MY_35/14.6.png'
actual_mask = load_actual_mask(actual_mask_path)
iou = calculate_iou(predicted_mask, actual_mask)
print(f"Intersection over Union (IoU): {iou:.3f}")

# Convert binary masks to uint8
predicted_mask_uint8 = (predicted_mask * 255).astype(np.uint8)
actual_mask_uint8 = (actual_mask * 255).astype(np.uint8)

# Find contours for predicted mask
contours_pred, hierarchy_pred = cv2.findContours(predicted_mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_png, contours_pred, -1, (255, 0, 0), 3)  # Red contours for predicted mask

# Find contours for actual mask
contours_act, hierarchy_act = cv2.findContours(actual_mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_png, contours_act, -1, (0, 255, 0), 3)  # Green contours for actual mask

# Display the comparison
plt.imshow(cv2.cvtColor(image_png, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
plt.title('Contour Overlays')
plt.axis('off')

plt.show()