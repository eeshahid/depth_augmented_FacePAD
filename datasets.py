


##################################################################################
############# Dataset Class -  Img (or Depth) > 3channel) for OULU_NPU ###########
##################################################################################
class VideoDataset3ch_OULU(Dataset):
    def __init__(self, orig_root_dir, file_list_path, transform=None, num_frames=16, is_train=False, protocol=None):
        """
        Args:
            orig_root_dir (str): Path to the root directory containing the original video files.
            depth_root_dir (str): Path to the root directory containing the corresponding depth video files.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int, optional): Number of frames to be sampled from each video.
            is_train (bool, optional): Flag to indicate if the dataset is used for training.
        """
        self.orig_root_dir = orig_root_dir
        self.file_list_path = file_list_path
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.classes = ['attack', 'real']  # Label mapping: 0 = attack, 1 = real
        self.protocol = protocol
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load paths to original videos and their corresponding depth videos, along with labels.
        """
        samples = []

        if self.protocol!='all':
            # Read the file list and extract filenames and labels
            with open(self.file_list_path, 'r') as f:
                lines = f.readlines()

            valid_files = {}  # Dictionary to store file names and labels
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue  # Skip malformed lines
                
                label = 1 if parts[0].strip() == "+1" else 0
                filename = parts[1].strip()
                valid_files[filename] = label  # Store without extension

            # Loop through files for the user (Phone_Session_User_File.avi)
            for file_name in os.listdir(self.orig_root_dir):
                if file_name.endswith(('.avi', '.mp4', '.mov')):
                    base_name, ext = os.path.splitext(file_name)  # Get filename without extension
                    if base_name in valid_files:

                        video_path = os.path.join(self.orig_root_dir, file_name)

                        # Extract access type from the file name
                        access_type = int(file_name.split('_')[-1].split('.')[0])

                        # Determine label based on access type (1 = real, 2-5 = attack)
                        label = 1 if access_type == 1 else 0

                        samples.append((video_path, label))

        else:
            print('All data being used... No OULU-NPU Protocol Applied\n')

            for file_name in os.listdir(self.orig_root_dir):
                if file_name.endswith(('.avi', '.mp4', '.mov')):
                    video_path = os.path.join(self.orig_root_dir, file_name)

                    # Extract access type from the file name
                    access_type = int(file_name.split('_')[-1].split('.')[0])

                    # Determine label based on access type (1 = real, 2-5 = attack)
                    label = 1 if access_type == 1 else 0

                    samples.append((video_path, label))
                        
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices, is_depth=False):
        """
        Load frames from the specified video file at the given frame indices.
        For depth videos, convert frames to single-channel grayscale.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:  # Ensure indices do not exceed total frames
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            if is_depth:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = np.expand_dims(frame, axis=-1)  # Add channel dimension

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if not is_depth else frame
            frames.append(F.to_tensor(frame))

        cap.release()

        # Handle the case where no frames were loaded
        if not frames:
            print(f"No frames loaded for video: {video_path}, indices: {frame_indices}")
            with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
                log_file.write(f"No frames loaded for video: {video_path}, indices: {frame_indices}")

            # Handle empty frames list by appending a black frame of the expected size
            placeholder_frame = np.zeros((224, 224, 1 if is_depth else 3), dtype=np.uint8)
            frames.append(F.to_tensor(placeholder_frame))

        # Pad if fewer frames are available
        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        """
        Override __getitem__ to load frames from both original and depth videos
        with synchronized indices.
        """
        # orig_video_path, depth_video_path, label = self.samples[idx]
        orig_video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(orig_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Determine frame indices
        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(start_frame, start_frame + self.num_frames - 1, self.num_frames, dtype=int)

        # Load frames from both videos using the same frame indices
        orig_frames = self._load_frames(orig_video_path, frame_indices, is_depth=False)
        # depth_frames = self._load_frames(depth_video_path, frame_indices, is_depth=True)        
        
        # Apply transformations to both original and depth frames
        if self.transform:
            orig_frames = [self.transform(frame) for frame in orig_frames]
            # depth_frames = [self.transform(frame) for frame in depth_frames]

        # Apply augmentation if in training mode
        if self.is_train:
            angle, scale = self._random_augmentation_params()
            orig_frames = [self.apply_augmentation(frame, angle, scale) for frame in orig_frames]
            # depth_frames = [self.apply_augmentation(frame, angle, scale) for frame in depth_frames]

        orig_frames = torch.stack(orig_frames)  # Shape: (num_frames, 3, H, W)
        # depth_frames = torch.stack(depth_frames)  # Shape: (num_frames, 1, H, W)

        # # Combine into a 4-channel tensor
        # combined_frames = torch.cat([orig_frames, depth_frames], dim=1)  # Shape: (num_frames, 4, H, W)
        combined_frames = orig_frames

        return combined_frames, label
        
        
    def _random_augmentation_params(self):
        """
        Generate random augmentation parameters (angle and scale) for training.
        """
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        """Apply rotation and scaling augmentation."""
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image

##################################################################################
############# Dataset Class -  Img+Depth > 4channel) for OULU_NPU ############
##################################################################################
class VideoDataset4ch_OULU(Dataset):
    def __init__(self, orig_root_dir, depth_root_dir, file_list_path, transform=None, num_frames=16, is_train=False, protocol=None):
        """
        Args:
            orig_root_dir (str): Path to the root directory containing the original video files.
            depth_root_dir (str): Path to the root directory containing the corresponding depth video files.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int, optional): Number of frames to be sampled from each video.
            is_train (bool, optional): Flag to indicate if the dataset is used for training.
        """
        self.orig_root_dir = orig_root_dir
        self.depth_root_dir = depth_root_dir
        self.file_list_path = file_list_path
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.classes = ['attack', 'real']  # Label mapping: 0 = attack, 1 = real
        self.protocol = protocol
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load paths to original videos and their corresponding depth videos, along with labels.
        """
        samples = []

        if self.protocol!='all':
            # Read the file list and extract filenames and labels
            with open(self.file_list_path, 'r') as f:
                lines = f.readlines()

            valid_files = {}  # Dictionary to store file names and labels
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue  # Skip malformed lines
                
                label = 1 if parts[0].strip() == "+1" else 0
                filename = parts[1].strip()
                valid_files[filename] = label  # Store without extension

            # Loop through files for the user (Phone_Session_User_File.avi)
            for file_name in os.listdir(self.orig_root_dir):
                if file_name.endswith(('.avi', '.mp4', '.mov')):
                    base_name, ext = os.path.splitext(file_name)  # Get filename without extension
                    if base_name in valid_files:

                        video_path = os.path.join(self.orig_root_dir, file_name)

                        # Extract access type from the file name
                        access_type = int(file_name.split('_')[-1].split('.')[0])

                        # Determine label based on access type (1 = real, 2-5 = attack)
                        label = 1 if access_type == 1 else 0

                        # Corresponding depth map video file
                        depth_video_path = video_path.replace(self.orig_root_dir, self.depth_root_dir)#.replace('.avi', '.mp4')

                        # Ensure corresponding depth video exists
                        if os.path.exists(depth_video_path):
                            samples.append((video_path, depth_video_path, label))
                        else:
                            print(f"Warning: Depth map video not found for {video_path}")                            
                            # samples.append((video_path, 'None', label))

        else:
            # Loop through files for the user (Phone_Session_User_File.avi)
            for file_name in os.listdir(self.orig_root_dir):
                if file_name.endswith(('.avi', '.mp4', '.mov')):
                    video_path = os.path.join(self.orig_root_dir, file_name)

                    # Extract access type from the file name
                    access_type = int(file_name.split('_')[-1].split('.')[0])

                    # Determine label based on access type (1 = real, 2-5 = attack)
                    label = 1 if access_type == 1 else 0

                    # Corresponding depth map video file
                    depth_video_path = video_path.replace(self.orig_root_dir, self.depth_root_dir)#.replace('.avi', '.mp4')

                    # Ensure corresponding depth video exists
                    if os.path.exists(depth_video_path):
                        samples.append((video_path, depth_video_path, label))
                    else:
                        print(f"Warning: Depth map video not found for {video_path}")                            
                        # samples.append((video_path, 'None', label))
                        
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices, is_depth=False):
        """
        Load frames from the specified video file at the given frame indices.
        For depth videos, convert frames to single-channel grayscale.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:  # Ensure indices do not exceed total frames
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            if is_depth:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = np.expand_dims(frame, axis=-1)  # Add channel dimension

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if not is_depth else frame
            frames.append(F.to_tensor(frame))

        cap.release()

        # Handle the case where no frames were loaded
        if not frames:
            print(f"No frames loaded for video: {video_path}, indices: {frame_indices}")
            with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
                log_file.write(f"No frames loaded for video: {video_path}, indices: {frame_indices}")

            # Handle empty frames list by appending a black frame of the expected size
            placeholder_frame = np.zeros((224, 224, 1 if is_depth else 3), dtype=np.uint8)
            frames.append(F.to_tensor(placeholder_frame))

        # Pad if fewer frames are available
        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        """
        Override __getitem__ to load frames from both original and depth videos
        with synchronized indices.
        """
        orig_video_path, depth_video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(orig_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Determine frame indices
        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(start_frame, start_frame + self.num_frames - 1, self.num_frames, dtype=int)

        # Load frames from both videos using the same frame indices
        orig_frames = self._load_frames(orig_video_path, frame_indices, is_depth=False)
        depth_frames = self._load_frames(depth_video_path, frame_indices, is_depth=True)        
        
        # Apply transformations to both original and depth frames
        if self.transform:
            orig_frames = [self.transform(frame) for frame in orig_frames]
            depth_frames = [self.transform(frame) for frame in depth_frames]

        # Apply augmentation if in training mode
        if self.is_train:
            angle, scale = self._random_augmentation_params()
            orig_frames = [self.apply_augmentation(frame, angle, scale) for frame in orig_frames]
            depth_frames = [self.apply_augmentation(frame, angle, scale) for frame in depth_frames]

        orig_frames = torch.stack(orig_frames)  # Shape: (num_frames, 3, H, W)
        depth_frames = torch.stack(depth_frames)  # Shape: (num_frames, 1, H, W)

        # Combine into a 4-channel tensor
        combined_frames = torch.cat([orig_frames, depth_frames], dim=1)  # Shape: (num_frames, 4, H, W)

        return combined_frames, label
        
        
    def _random_augmentation_params(self):
        """
        Generate random augmentation parameters (angle and scale) for training.
        """
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        """Apply rotation and scaling augmentation."""
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image

######################################################
####### Dataset Class -  Img/Depth > 3channel) #######
######################################################
class VideoDataset3ch(Dataset):
    def __init__(self, orig_root_dir, transform=None, num_frames=16, is_train=False):
        """
        Args:
            orig_root_dir (str): Path to the root directory containing the original video files.
            depth_root_dir (str): Path to the root directory containing the corresponding depth video files.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int, optional): Number of frames to be sampled from each video.
            is_train (bool, optional): Flag to indicate if the dataset is used for training.
        """
        self.orig_root_dir = orig_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.classes = ['attack', 'real']  # Label mapping: 0 = attack, 1 = real
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load paths to original videos and their corresponding depth videos, along with labels.
        """
        samples = []
        for cls in self.classes:
            orig_cls_dir = os.path.join(self.orig_root_dir, cls)
            for root, _, files in os.walk(orig_cls_dir):
                for fname in files:
                    if fname.endswith(('.mp4', '.mov', '.avi')):
                        orig_video_path = os.path.join(root, fname)
                        samples.append((orig_video_path, self.classes.index(cls)))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices, is_depth=False):
        """
        Load frames from the specified video file at the given frame indices.
        For depth videos, convert frames to single-channel grayscale.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:  # Ensure indices do not exceed total frames
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            if is_depth:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = np.expand_dims(frame, axis=-1)  # Add channel dimension

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if not is_depth else frame
            frames.append(F.to_tensor(frame))

        cap.release()

        # Handle the case where no frames were loaded
        if not frames:
            print(f"No frames loaded for video: {video_path}, indices: {frame_indices}")
            with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
                log_file.write(f"No frames loaded for video: {video_path}, indices: {frame_indices}")

            # Handle empty frames list by appending a black frame of the expected size
            placeholder_frame = np.zeros((224, 224, 1 if is_depth else 3), dtype=np.uint8)
            frames.append(F.to_tensor(placeholder_frame))

        # Pad if fewer frames are available
        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        """
        Override __getitem__ to load frames from both original and depth videos
        with synchronized indices.
        """
        # orig_video_path, depth_video_path, label = self.samples[idx]
        orig_video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(orig_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Determine frame indices
        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(start_frame, start_frame + self.num_frames - 1, self.num_frames, dtype=int)

        # Load frames from both videos using the same frame indices
        orig_frames = self._load_frames(orig_video_path, frame_indices, is_depth=False)
        # depth_frames = self._load_frames(depth_video_path, frame_indices, is_depth=True)        
        
        # Apply transformations to both original and depth frames
        if self.transform:
            orig_frames = [self.transform(frame) for frame in orig_frames]
            # depth_frames = [self.transform(frame) for frame in depth_frames]

        # Apply augmentation if in training mode
        if self.is_train:
            angle, scale = self._random_augmentation_params()
            orig_frames = [self.apply_augmentation(frame, angle, scale) for frame in orig_frames]
            # depth_frames = [self.apply_augmentation(frame, angle, scale) for frame in depth_frames]

        orig_frames = torch.stack(orig_frames)  # Shape: (num_frames, 3, H, W)
        # depth_frames = torch.stack(depth_frames)  # Shape: (num_frames, 1, H, W)

        # # Combine into a 4-channel tensor
        # combined_frames = torch.cat([orig_frames, depth_frames], dim=1)  # Shape: (num_frames, 4, H, W)
        combined_frames = orig_frames

        return combined_frames, label
        
        
    def _random_augmentation_params(self):
        """
        Generate random augmentation parameters (angle and scale) for training.
        """
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        """Apply rotation and scaling augmentation."""
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image

######################################################
####### Dataset Class -  Img+Depth > 4channel) #######
######################################################
class VideoDataset4ch(Dataset):
    def __init__(self, orig_root_dir, depth_root_dir, transform=None, num_frames=16, is_train=False):
        self.orig_root_dir = orig_root_dir
        self.depth_root_dir = depth_root_dir
        self.transform = transform
        self.classes = ['attack', 'real']
        self.samples = self._load_samples()
        self.num_frames = num_frames
        self.is_train = is_train

    def _load_samples(self):
        """
        Load paths to original videos and their corresponding depth videos, along with labels.
        """
        samples = []
        for cls in self.classes:
            orig_cls_dir = os.path.join(self.orig_root_dir, cls)
            for root, _, files in os.walk(orig_cls_dir):
                for fname in files:
                    if fname.endswith(('.mp4', '.mov', '.avi')):
                        orig_video_path = os.path.join(root, fname)
                                                
                        depth_video_path = orig_video_path.replace(self.orig_root_dir, self.depth_root_dir)
                        depth_video_path = os.path.splitext(depth_video_path)[0] + '.mp4'

                        if os.path.exists(depth_video_path):
                            samples.append((orig_video_path, depth_video_path, self.classes.index(cls)))

                        else:
                            print(f"Warning: Depth map video not found for {orig_video_path}")                            
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices, is_depth=False):
        """
        Load frames from the specified video file at the given frame indices.
        For depth videos, convert frames to single-channel grayscale.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:  # Ensure indices do not exceed total frames
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            if is_depth:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = np.expand_dims(frame, axis=-1)  # Add channel dimension

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if not is_depth else frame
            frames.append(F.to_tensor(frame))

        cap.release()

        # Handle the case where no frames were loaded
        if not frames:
            print(f"No frames loaded for video: {video_path}, indices: {frame_indices}")
            with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
                log_file.write(f"No frames loaded for video: {video_path}, indices: {frame_indices}")

            # Handle empty frames list by appending a black frame of the expected size
            placeholder_frame = np.zeros((224, 224, 1 if is_depth else 3), dtype=np.uint8)
            frames.append(F.to_tensor(placeholder_frame))

        # Pad if fewer frames are available
        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        """
        Override __getitem__ to load frames from both original and depth videos
        with synchronized indices.
        """
        orig_video_path, depth_video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(orig_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Determine frame indices
        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(start_frame, start_frame + self.num_frames - 1, self.num_frames, dtype=int)

        # Load frames from both videos using the same frame indices
        orig_frames = self._load_frames(orig_video_path, frame_indices, is_depth=False)
        depth_frames = self._load_frames(depth_video_path, frame_indices, is_depth=True)        
        
        # Apply transformations to both original and depth frames
        if self.transform:
            orig_frames = [self.transform(frame) for frame in orig_frames]
            depth_frames = [self.transform(frame) for frame in depth_frames]

        # Apply augmentation if in training mode
        if self.is_train:
            angle, scale = self._random_augmentation_params()
            orig_frames = [self.apply_augmentation(frame, angle, scale) for frame in orig_frames]
            depth_frames = [self.apply_augmentation(frame, angle, scale) for frame in depth_frames]

        orig_frames = torch.stack(orig_frames)  # Shape: (num_frames, 3, H, W)
        depth_frames = torch.stack(depth_frames)  # Shape: (num_frames, 1, H, W)

        # Combine into a 4-channel tensor
        combined_frames = torch.cat([orig_frames, depth_frames], dim=1)  # Shape: (num_frames, 4, H, W)

        return combined_frames, label
        
        
    def _random_augmentation_params(self):
        """
        Generate random augmentation parameters (angle and scale) for training.
        """
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        """Apply rotation and scaling augmentation."""
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image


