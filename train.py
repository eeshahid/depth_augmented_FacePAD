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

from models import DualBranchMobileNet, MultiChannelCNN, DepthAuxiliaryMobileNet, CNN3ch, MobileNetStudent
from datasets import VideoDataset3ch_OULU, VideoDataset4ch_OULU, VideoDataset3ch, VideoDataset4ch
from utils import create_log_directory, AdaptiveCenterCropAndResize, collate_fn, save_checkpoint, train_epoch, validate_epoch, train_epoch_distill, validate_epoch_student

############### Args ##########################
log_dir = create_log_directory(base_dir='/data/muhammad_jabbar/github/depth_augmented_FacePAD/logs')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Model with RGB + Depth input')
    parser.add_argument('--log_dir', type=str, default='/data/muhammad_jabbar/github/depth_augmented_FacePAD/logs', help='Path to save training logs and checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    parser.add_argument('--num_epochs', type=int, default=100, help='num_epochs for training the model (early stopping applied)')
    parser.add_argument('--num_frames', type=int, choices = [1], default=1, help='Number of frames for each video')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--model', type=str, default='DB', choices = ['3ch', 'DB, 4ch, Aux, KD'], help='Model to be trained')
    parser.add_argument('--chkpt_path', type=str, default='', help='Teacher model checkpoint path (.pth) for knowledge distillation (model=KD)')
    parser.add_argument('--orig_dataset_path', type=str, default='/data/muhammad_jabbar/datasets/Oulu_NPU', help='Path to original video dataset')
    parser.add_argument('--depth_dataset_path', type=str, default='/data/muhammad_jabbar/datasets/Oulu_NPU_depth_mp4', help='Path to cached depth map dataset (videos)')
    parser.add_argument('--dataset', type=str, default='RA', choices = ['RA, RM, RY, OULU'], help='Dataset to be used')
    parser.add_argument('--protocol', type=str, default='4', help='OULU-NPU dataset protocol being trained')
    parser.add_argument('--n_split', type=str, default='6', help='OULU-NPU dataset split for protocol 3 & 4')
    parser.add_argument('--exp_description', type=str, default='Train Model with RGB+Depth 4channel input, channel expansion, MobileNetv3Large (With pretrained weights), OULU-NPU train data, Protocol-4 (Split-6)')
    return parser.parse_args()
args = parse_args()

log_dir = create_log_directory(base_dir=args.log_dir)

# Print all arguments
print('\n############### Args ##########################')
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print('################################################\n')

# Print arguments to log file
with open(os.path.join(args.log_dir, 'training_log.txt'), 'w') as log_file:
    log_file.write('############### Args ##########################\n')
    for arg, value in vars(args).items():
        log_file.write(f"{arg}: {value}\n")
    log_file.write('################################################\n')
    log_file.write('\n')
######################################################

transform = transforms.Compose([AdaptiveCenterCropAndResize((224, 224))]),  # Adaptive crop, resize, and convert to tensor

if args.model == '3ch':
    if args.dataset == 'OULU-NPU':
        if args.protocol=='3' or args.protocol=='4':
            t_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Train_{args.n_split}.txt'
        else:
            t_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Train.txt'
        train_dataset = VideoDataset3ch_OULU(
            orig_root_dir=args.orig_dataset_path+'/Train_files',
            file_list_path = t_protocol_flist,
            transform=transform,
            num_frames=args.num_frames,
            is_train=True,
            protocol = args.protocol
        )
        if args.protocol=='3' or args.protocol=='4':
            v_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Dev_{args.n_split}.txt'
        else:
            v_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Dev.txt'
        val_dataset = VideoDataset3ch_OULU(
            orig_root_dir=args.orig_dataset_path+'/Dev_files',
            file_list_path = protocol_flist,
            transform=transform,
            num_frames=args.num_frames,
            is_train=False,
            protocol = args.protocol
        )
    else:
        train_dataset = VideoDataset3ch(
            orig_root_dir=args.orig_dataset_path+'/train',
            depth_root_dir=args.depth_dataset_path+'/train',
            transform=transform,
            num_frames=args.num_frames,
            is_train=True
        )
        val_dataset = VideoDataset3ch(
            orig_root_dir=args.orig_dataset_path+'/devel',
            depth_root_dir=args.depth_dataset_path+'/devel',
            transform=transform,
            num_frames=args.num_frames,
            is_train=False
        )
else: # for depth augmented models
    if args.dataset == 'OULU-NPU':
        if args.protocol=='3' or args.protocol=='4':
            t_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Train_{args.n_split}.txt'
        else:
            t_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Train.txt'
        train_dataset = VideoDataset4ch_OULU(
            orig_root_dir=args.orig_dataset_path+'/Train_files',
            depth_root_dir=args.depth_dataset_path+'/Train_files',
            file_list_path = t_protocol_flist,
            transform=transform,
            num_frames=args.num_frames,
            is_train=True,
            protocol = args.protocol
        )
        if args.protocol=='3' or args.protocol=='4':
            v_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Dev_{args.n_split}.txt'
        else:
            v_protocol_flist = f'/data/muhammad_jabbar/datasets/Oulu_NPU/Baseline/Protocol_{args.protocol}/Dev.txt'
        val_dataset = VideoDataset4ch_OULU(
            orig_root_dir=args.orig_dataset_path+'/Dev_files',
            depth_root_dir=args.depth_dataset_path+'/Dev_files',
            file_list_path = v_protocol_flist,
            transform=transform,
            num_frames=args.num_frames,
            is_train=False,
            protocol = args.protocol
        )
    else:
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
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
print(f'Train data loader length: {len(train_loader)}, Val data loader length: {len(val_loader)}\n')

########################## Model Instantiation ##########################

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

if args.model = 'DB':
    model = DualBranchMobileNet(num_classes=2).to(device)
elif args.model = 'Aux':
    model = DepthAuxiliaryMobileNet(num_classes=2).to(device)
elif args.model = '4ch':
    model = MultiChannelCNN(num_classes=2).to(device)
elif args.model = '3ch':
    model = CNN3ch(num_classes=2).to(device)
elif args.model = 'KD':
    student_model = MobileNetStudent(num_classes=2).to(device)
    checkpoint = torch.load(args.chkpt_path, weights_only=True)
    teacher_model = DualBranchMobileNet(num_classes=2).to(device)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
else:
    raise ValueError("Invalid model choice. Choose from ['3ch', 'DB', '4ch', 'Aux', 'KD'].")

########################## Train ##########################

# 
if args.model != 'KD':
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
else:
    # Distillation hyperparameters
    alpha = 0.7  # weight for distillation loss
    temperature = 3.0
    criterion_CE = nn.CrossEntropyLoss()
    criterion_KL = nn.KLDivLoss(reduction='batchmean')
    optimizer_student = optim.Adam(student_model.parameters(), lr=1e-4)

writer = SummaryWriter(log_dir=args.log_dir)
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints') # Directory to save checkpoints
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Initialize variables to track best validation loss
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 10  # Number of epochs to wait before stopping early

# Training loop
num_epochs = args.num_epochs
for epoch in range(num_epochs):

    # Train and validate for each epoch
    if args.model == 'KD':
        train_loss, train_acc = train_epoch_distill(
            student_model, teacher_model, train_loader, criterion_CE, criterion_KL, optimizer_student,
            alpha, temperature, epoch
        )
        val_loss, val_acc = validate_epoch_student(student_model, val_loader, criterion_CE)
    else:
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

    # Logging metrics
    with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
        log_file.write(f"Epoch {epoch+1}, Training Loss: {train_loss}, Training Acc: {train_acc}\n")
        log_file.write(f"Epoch {epoch+1}, Valication Loss: {val_loss}, Valication Acc: {val_acc}\n")
        log_file.write("\n")    
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


