



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


# Function to save model checkpoints
def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best_model.pth"))


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


def train_epoch_distill(student, teacher, loader, criterion_CE, criterion_KL, optimizer, alpha, temperature, epoch):
    student.train()
    running_loss, correct, total = 0.0, 0, 0
    
    with tqdm(loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Teacher predictions (no grad)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # Student predictions (only RGB channels)
            rgb_inputs = inputs[:, :, :3, :, :]  # [B, num_frames, 3, H, W]
            student_logits = student(rgb_inputs)
            
            # Compute losses
            loss_ce = criterion_CE(student_logits, labels)
            loss_kd = criterion_KL(
                F_nn.log_softmax(student_logits / temperature, dim=1),
                F_nn.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature ** 2)

            loss = alpha * loss_kd + (1 - alpha) * loss_ce

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(student_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            tepoch.set_postfix(loss=running_loss/total, accuracy=100. * correct/total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation loop
def validate_epoch_student(student, loader, criterion):
    student.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            rgb_inputs = inputs[:, :, :3, :, :]
            logits = student(rgb_inputs)

            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


