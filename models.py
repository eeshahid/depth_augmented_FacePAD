import torch
import torch.nn as nn
from torchvision import models

################################################
##### Model Class (Image/Depth, 3-channels) ####
################################################

class CNN3ch(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN3ch, self).__init__()

        # Use MobileNetV3 as the CNN backbone
        # self.cnn = models.mobilenet_v3_large(pretrained=True)
        self.cnn = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # self.cnn = models.mobilenet_v3_large(weights=None)
        self.cnn.classifier = nn.Identity()  # Remove the classifier to use the feature extractor
        self.fc = nn.Linear(960, num_classes)

        # # Use ResNet-101 as the CNN backbone
        # self.cnn = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # self.cnn.fc = nn.Identity()  # Remove the classifier to use the feature extractor
        # self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()  # Expect input as [batch_size, seq_len, channels, height, width]
        out_decision_t = []
        cnn_features = []
        
        for t in range(seq_len):
            # with torch.no_grad():
            feature = self.cnn(x[:, t, :, :, :])  # Extract CNN features for each frame
            cnn_features.append(feature)
            # out_decision_t.append(self.fc(feature))
        
        # Stack features from all frames and average over the temporal dimension (seq_len)
        # out_decision_t = torch.stack(out_decision_t, dim=1)  # Shape: [batch_size, seq_len, 960]
        # out = out_decision_t.mean(dim=1)  # Temporal average pooling: Shape: [batch_size, 960]

        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: [batch_size, seq_len, 960]
        temporal_avg_features = cnn_features.mean(dim=1)  # Temporal average pooling: Shape: [batch_size, 960]

        # Pass through the final fully connected layer
        out = self.fc(temporal_avg_features)  # Shape: [batch_size, num_classes]
        return out

    def extract_intermediate_features(self, x):
        """Extract features before and after LSTM."""
        batch_size, seq_len, C, H, W = x.size()
        cnn_features = []
        for t in range(seq_len):
            # with torch.no_grad():
            feature = self.cnn(x[:, t, :, :, :])  # CNN output (before LSTM)
            cnn_features.append(feature)
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: [batch_size, seq_len, cnn_output_dim]
        temporal_avg_features = cnn_features.mean(dim=1)  # Temporal average pooling: Shape: [batch_size, 960]
        return cnn_features, temporal_avg_features

################################################
######## Image-Depth Hybrid Models #############
################################################

# 1. Multi-Channel Input with Channel Expansion
# Input: 224*224*4, image + depth_map
# Modified first Conv2d layer to take 4 input channels

class MultiChannelCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiChannelCNN, self).__init__()
        
        # Load MobileNetV3 as the CNN backbone
        mobilenet_v3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 4 input channels (RGB + Depth)
        original_conv1 = mobilenet_v3.features[0][0]  # Access the first Conv2d layer
        self.custom_conv1 = nn.Conv2d(
            in_channels=4,  # Change input channels to 4
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Copy the weights of the original 3-channel convolution to the first 3 channels
        with torch.no_grad():
            self.custom_conv1.weight[:, :3, :, :] = original_conv1.weight
            self.custom_conv1.weight[:, 3:, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Replace the original first layer in MobileNetV3
        mobilenet_v3.features[0][0] = self.custom_conv1
        
        # Remove the classifier to use MobileNetV3 as a feature extractor
        self.cnn = mobilenet_v3
        self.cnn.classifier = nn.Identity()  # No classification head
        
        # Fully connected layer for the final classification
        self.fc = nn.Linear(960, num_classes)  # 960 is the output feature size of MobileNetV3

    def forward(self, x):
        """
        Forward pass for multi-channel input.
        x_image: Tensor of shape [batch_size, seq_len, 3, 224, 224] (RGB frames)
        x_depth: Tensor of shape [batch_size, seq_len, 1, 224, 224] (Depth frames)
        """
        batch_size, seq_len, _, H, W = x.size()
        
        # # Concatenate RGB and Depth channels along the channel dimension
        # x = torch.cat([x_image, x_depth], dim=2)  # Shape: [batch_size, seq_len, 4, 224, 224]
        
        cnn_features = []
        for t in range(seq_len):
            feature = self.cnn(x[:, t, :, :, :])  # Extract features for each frame
            cnn_features.append(feature)
        
        # Temporal average pooling
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: [batch_size, seq_len, 960]
        temporal_avg_features = cnn_features.mean(dim=1)  # Shape: [batch_size, 960]

        # Pass through the final fully connected layer
        out = self.fc(temporal_avg_features)  # Shape: [batch_size, num_classes]
        return out

################################################

# 2. Separate Branches with Feature Concatenation
# Input: 224*224*4, image + depth_map
# RGB branch MobileNet with 3-channel input
# Depth map branch MobileNet with 1-channel input
# Output features from both branches are concatenated, and passed to classifier

class DualBranchMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DualBranchMobileNet, self).__init__()
        
        # RGB branch: Pretrained MobileNetV3 for RGB input (3 channels)
        self.rgb_branch = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # Modify classifier for the RGB branch
        rgb_features_in = self.rgb_branch.classifier[0].in_features
        self.rgb_branch.classifier = nn.Identity()  # Remove classifier, keep feature extractor
        
        # Depth branch: Another MobileNetV3 for depth map (1 channel)
        self.depth_branch = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # Modify the first layer of the depth branch to accept 1-channel input
        self.depth_branch.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.depth_branch.classifier = nn.Identity()  # Remove classifier for feature extraction

        # Concatenate the output of both branches and pass to the classifier
        concat_feature_size = rgb_features_in * 2  # Combine features from both branches
        
        # Final classifier (after feature concatenation)
        self.classifier = nn.Sequential(
            nn.Linear(concat_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch_size, num_frames=1, channels=4, height=224, width=224]
        batch_size, num_frames, channels, height, width = x.shape
        assert num_frames == 1, "This model supports single-frame input only."
        
        # Squeeze the frame dimension
        x = x.squeeze(1)  # Shape: [batch_size, channels=4, height, width]
        
        # Split the input into RGB (3 channels) and depth (1 channel)
        rgb_input = x[:, :3, :, :]  # First 3 channels for RGB
        depth_input = x[:, 3:, :, :]  # Last channel for depth map
        
        # Pass through the RGB branch
        rgb_features = self.rgb_branch(rgb_input)  # Shape: [batch_size, rgb_features_in]
        
        # Pass through the depth branch
        depth_features = self.depth_branch(depth_input)  # Shape: [batch_size, rgb_features_in]
        
        # Concatenate the features from both branches
        combined_features = torch.cat((rgb_features, depth_features), dim=1)  # Shape: [batch_size, concat_feature_size]
        
        # Pass through the final classifier
        output = self.classifier(combined_features)  # Shape: [batch_size, num_classes]
        
        return output

################################################

# 3. Depth as Auxiliary Input
# Input: 224*224*4, image + depth_map
# MobileNet with 3-channel RGB input
# Simple CNN for depth features
# RGB features before MobileNet final classification layer are modulated, by depth features

class DepthAuxiliaryMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DepthAuxiliaryMobileNet, self).__init__()
        
        # RGB branch: Pretrained MobileNetV3 for RGB input (3 channels)
        self.rgb_branch = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        rgb_features_in = self.rgb_branch.classifier[0].in_features
        self.rgb_branch.classifier = nn.Identity()  # Remove the classification head
        
        # Depth branch: Simple CNN for extracting depth features
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.depth_feature_dim = 32  # Output size from the depth branch
        
        # Depth modulation layer to integrate depth features with RGB features
        self.modulation_layer = nn.Sequential(
            nn.Linear(self.depth_feature_dim, rgb_features_in),
            nn.Sigmoid()  # Output range [0, 1] for modulation
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(rgb_features_in, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch_size, num_frames=1, channels=4, height=224, width=224]
        batch_size, num_frames, channels, height, width = x.shape
        assert num_frames == 1, "This model supports single-frame input only."
        
        # Squeeze the frame dimension
        x = x.squeeze(1)  # Shape: [batch_size, channels=4, height, width]
        
        # Split the input into RGB (3 channels) and depth (1 channel)
        rgb_input = x[:, :3, :, :]  # First 3 channels for RGB
        depth_input = x[:, 3:, :, :]  # Last channel for depth map
        
        # Pass through the RGB branch
        rgb_features = self.rgb_branch(rgb_input)  # Shape: [batch_size, rgb_features_in]
        
        # Pass through the depth branch
        depth_features = self.depth_branch(depth_input)  # Shape: [batch_size, 32, 1, 1]
        depth_features = depth_features.view(batch_size, -1)  # Flatten: [batch_size, 32]
        
        # Modulate RGB features with depth features
        modulation_weights = self.modulation_layer(depth_features)  # Shape: [batch_size, rgb_features_in]
        modulated_rgb_features = rgb_features * modulation_weights  # Element-wise multiplication
        
        # Pass through the final classifier
        output = self.classifier(modulated_rgb_features)  # Shape: [batch_size, num_classes]
        
        return output

################################################
############ Student  Model ####################
################################################
class MobileNetStudent(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetStudent, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.feature_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, num_frames=1, channels=3, height, width]
        x = x.squeeze(1)
        features = self.backbone(x)
        output = self.classifier(features)
        return output
