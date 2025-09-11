## Batch processing

import os
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
from tqdm import tqdm

# Initialize depth estimation model
device = 0  # Set to GPU device
depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=device)

# Define directories and splits
dataset_path = '/data/muhammad_jabbar/datasets/Oulu_NPU'
output_dir = '/data/muhammad_jabbar/datasets/Oulu_NPU_depth_mp4'
splits = ['Train_files', 'Dev_files', 'Test_files']
batch_size = 64  # Adjust batch size based on GPU memory
failed_videos = []  # List to store failed video files

# Traverse all subdirectories and process videos
for split in splits:
    split_input_dir = os.path.join(dataset_path, split)
    split_output_dir = os.path.join(output_dir, split)

    for root, _, files in os.walk(split_input_dir):
        # Determine relative path to preserve directory structure
        relative_path = os.path.relpath(root, split_input_dir)
        output_subdir = os.path.join(split_output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        for video_name in tqdm(files, desc=f"Processing videos in {split}/{relative_path}"):
            if not video_name.endswith(('.mp4', '.mov', '.avi')):
                continue

            input_video_path = os.path.join(root, video_name)
            output_video_path = os.path.join(output_subdir, video_name)

            try:
                # Capture the video
                cap = cv2.VideoCapture(input_video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Define video writer for output
                out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)

                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert frame to PIL Image and add to batch
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))

                    # Process batch if it reaches the batch_size
                    if len(frames) == batch_size:
                        depth_maps = depth_model(frames)  # Batch process frames
                        for depth in depth_maps:
                            depth_array = np.array(depth['depth'])
                            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                            depth_grayscale = depth_normalized.astype(np.uint8)  # Grayscale depth
                            out.write(depth_grayscale)
                        frames = []  # Clear frames after processing

                # Process any remaining frames at the end of video capture
                if frames:
                    depth_maps = depth_model(frames)
                    for depth in depth_maps:
                        depth_array = np.array(depth['depth'])
                        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                        depth_grayscale = depth_normalized.astype(np.uint8)  # Grayscale depth
                        out.write(depth_grayscale)

                # Release resources
                cap.release()
                out.release()

            except Exception as e:
                print(f"Failed to process {video_name}: {e}")
                failed_videos.append(input_video_path)

# Save failed videos list to a text file in output_dir
if failed_videos:
    failed_videos_path = os.path.join(output_dir, 'failed_videos.txt')
    with open(failed_videos_path, 'w') as f:
        for failed_video in failed_videos:
            f.write(f"{failed_video}\n")
    print(f"\nThe list of failed videos has been saved to: {failed_videos_path}")
else:
    print("All videos processed successfully.")
