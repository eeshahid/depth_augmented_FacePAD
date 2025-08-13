import os
import argparse
from datetime import datetime
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import functional as F
import random
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

###############################################
############### Args ##########################
###############################################

def create_log_directory(base_dir='/home/muhammad_jabbar/face_PAD/logs'):
    # Ensure the base log directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Get a list of existing directories in the base_dir
    existing_dirs = os.listdir(base_dir)
    # Find the highest current log number
    log_numbers = []
    for d in existing_dirs:
        if d.startswith('log_'):
            try:
                # Extract the numeric part and check for format
                num_str = d.split('_')[1]
                log_numbers.append(int(num_str))
            except ValueError:
                pass
    # Determine the next log number
    next_num = 1 if not log_numbers else max(log_numbers) + 1
    log_num = f"{next_num:03d}"  # Ensure a 3-digit number
    # Get current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the new log directory name
    new_log_dir = os.path.join(base_dir, f'log_{log_num}_{timestamp}')
    # Create the new directory
    os.makedirs(new_log_dir)
    return new_log_dir

log_dir = create_log_directory()
#############

def parse_args():
    parser = argparse.ArgumentParser(description='Train Model with RGB + Depth input')

    # Adding arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    parser.add_argument('--num_frames', type=int, default=1, help='Number of frames for each video')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--orig_dataset_path', type=str, default='/data/muhammad_jabbar/datasets/Oulu_NPU', help='Path to original video dataset')
    parser.add_argument('--depth_dataset_path', type=str, default='/data/muhammad_jabbar/datasets/Oulu_NPU_depth_mp4', help='Path to depth map dataset')
    parser.add_argument('--protocol', type=str, default='4', help='OULU-NPU dataset protocol being trained')
    parser.add_argument('--n_split', type=str, default='6', help='OULU-NPU dataset split for protocol 3 & 4')

    parser.add_argument('--exp_description', type=str, default='Train Model with RGB+Depth 4channel input, channel expansion, MobileNetv3Large (With pretrained weights), OULU-NPU train data, Protocol-4 (Split-6)')

    return parser.parse_args()

args = parse_args()
args.log_dir = log_dir

# Print all arguments
print()
print('############### Args ##########################')
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print('################################################')
print()

# Print arguments to log file
with open(os.path.join(args.log_dir, 'training_log.txt'), 'w') as log_file:
    log_file.write('############### Args ##########################\n')
    for arg, value in vars(args).items():
        log_file.write(f"{arg}: {value}\n")
    log_file.write('################################################\n')
    log_file.write('\n')

######################################################

class AdaptiveCenterCropAndResize:
    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): The desired output size after resizing (e.g., (32, 32)).
        """
        self.output_size = output_size
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        # Convert tensor to PIL image if necessary
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        # Handle single-channel images
        if img.mode != 'RGB':
            img = img.convert('L')  # Convert to grayscale mode
            
        # Get image size (width, height)
        width, height = img.size

        # Find the minimum dimension to create the largest possible square
        crop_size = min(width, height)

        # Calculate the coordinates to center-crop the square
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2

        # Crop the image to the largest square
        img = img.crop((left, top, right, bottom))

        # Resize the cropped square to the desired output size
        img = img.resize(self.output_size, Image.Resampling.LANCZOS)

        # Convert the resized image back to a tensor
        img = self.to_tensor(img)

        return img

def collate_fn(batch):
    max_length = max([frames.size(0) for frames, _ in batch])  # Get the maximum sequence length
    padded_frames = []  # To store padded 4-channel tensors
    labels = []  # To store labels
    
    for frames, label in batch:
        if frames.size(0) < max_length:
            # Pad with zeros along the frame dimension
            padding = torch.zeros((max_length - frames.size(0), *frames.shape[1:]))
            # padded_frames.append(torch.cat((frames, padding), dim=0))
            padded_frames.append(torch.cat((frames, padding), dim=0))  # Pad at the end
        else:
            padded_frames.append(frames)

        labels.append(label)
    
    # Stack all sequences and labels
    padded_frames = torch.stack(padded_frames)  # Shape: (batch_size, max_length, 4, H, W)
    labels = torch.tensor(labels)  # Shape: (batch_size,)

    return padded_frames, labels

######################################################

transform = transforms.Compose([
    AdaptiveCenterCropAndResize((224, 224)),  # Adaptive crop, resize, and convert to tensor
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
])

train_dataset = VideoDataset4ch(
    orig_root_dir=args.orig_dataset_path+'/train',
    depth_root_dir=args.depth_dataset_path+'/train',
    transform=transform,
    num_frames=args.num_frames,
    is_train=True
)

val_dataset = VideoDataset4ch(
    orig_root_dir=args.orig_dataset_path+'/devel',
    depth_root_dir=args.depth_dataset_path+'/devel',
    transform=transform,
    num_frames=args.num_frames,
    is_train=False
)

# Instantiate dataloaders with custom collate function
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print()

print(f'Train data loader length: {len(train_loader)}')
print(f'Val data loader length: {len(val_loader)}')
print()

############### Instantiate the model ##########################

device = torch.device(f'cuda:{args.gpu}')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = DualBranchMobileNet(num_classes=2).to(device)

########################## Train ##########################

# Hyperparameters and setup
criterion = nn.CrossEntropyLoss()  # For final classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)
writer = SummaryWriter(log_dir=args.log_dir)

# Directory to save checkpoints
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Function to save model checkpoints
def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best_model.pth"))

# Training loop
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Supervised Contrastive Learning (or CrossEntropy)
            loss = criterion(outputs, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            tepoch.set_postfix(loss=running_loss/total, accuracy=100. * correct/total)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation loop
def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Loss calculation
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Initialize variables to track best validation loss
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 10  # Number of epochs to wait before stopping early

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Train and validate for each epoch
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)

    with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
        log_file.write(f"Epoch {epoch+1}, Training Loss: {train_loss}, Training Acc: {train_acc}\n")
        log_file.write(f"Epoch {epoch+1}, Valication Loss: {val_loss}, Valication Acc: {val_acc}\n")
        log_file.write("\n")
    
    # Logging metrics
    writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
    writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save best model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0  # Reset early stopping counter
        is_best = True
    else:
        early_stopping_counter += 1
        is_best = False

    # Save current checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, is_best=is_best, filename=f"checkpoint_epoch_{epoch + 1}.pth")

    # Early stopping condition
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered")

        with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
            log_file.write("Early stopping triggered")

        break

writer.close()


