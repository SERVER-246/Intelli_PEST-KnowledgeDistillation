"""
Student Model Architecture for Knowledge Distillation
=====================================================
Custom lightweight CNN designed for pest classification.
Target: <20MB model size with high accuracy.

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for parameter efficiency."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    """Standard Convolution Block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        use_depthwise: bool = False
    ):
        super().__init__()
        
        if use_depthwise:
            self.conv = DepthwiseSeparableConv(
                in_channels, out_channels, kernel_size, stride, padding
            )
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm)
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual Block with optional SE attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        use_depthwise: bool = True
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, use_depthwise=use_depthwise
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, use_depthwise=use_depthwise
        )
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        return F.relu(out, inplace=True)


class StudentCNN(nn.Module):
    """
    Lightweight Student CNN for Knowledge Distillation.
    
    Architecture designed for:
    - Small model size (<20MB)
    - Fast inference
    - High accuracy through knowledge distillation
    
    Features:
    - Depthwise separable convolutions
    - Squeeze-and-Excitation blocks
    - Residual connections
    - Global average pooling
    """
    
    def __init__(
        self,
        num_classes: int = 12,
        input_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256, 384],
        dropout_rate: float = 0.3,
        use_depthwise: bool = True,
        use_se: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = channels[-1]
        
        # Initial convolution (standard conv for better feature extraction)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction stages
        self.stage1 = self._make_stage(channels[0], channels[1], num_blocks=2, stride=2, use_depthwise=use_depthwise, use_se=use_se)
        self.stage2 = self._make_stage(channels[1], channels[2], num_blocks=3, stride=2, use_depthwise=use_depthwise, use_se=use_se)
        self.stage3 = self._make_stage(channels[2], channels[3], num_blocks=3, stride=2, use_depthwise=use_depthwise, use_se=use_se)
        self.stage4 = self._make_stage(channels[3], channels[4], num_blocks=2, stride=2, use_depthwise=use_depthwise, use_se=use_se)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(channels[4], channels[4] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(channels[4] // 2, num_classes)
        )
        
        # Feature hooks for knowledge distillation
        self.intermediate_features = {}
        
        # Initialize weights
        self._initialize_weights()
        
        # Log model info
        self._log_model_info()
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_depthwise: bool,
        use_se: bool
    ) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(
            in_channels, out_channels, stride=stride,
            use_se=use_se, use_depthwise=use_depthwise
        ))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(
                out_channels, out_channels, stride=1,
                use_se=use_se, use_depthwise=use_depthwise
            ))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _log_model_info(self):
        """Log model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        logger.info(f"StudentCNN Architecture:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Estimated model size: {model_size_mb:.2f} MB")
        logger.info(f"  - Number of classes: {self.num_classes}")
    
    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features at multiple levels for knowledge distillation.
        
        Returns:
            features: Final feature vector before classifier
            intermediate: List of intermediate feature maps
        """
        intermediate = []
        
        x = self.stem(x)
        intermediate.append(x)
        
        x = self.stage1(x)
        intermediate.append(x)
        
        x = self.stage2(x)
        intermediate.append(x)
        
        x = self.stage3(x)
        intermediate.append(x)
        
        x = self.stage4(x)
        intermediate.append(x)
        
        features = self.global_pool(x).flatten(1)
        
        return features, intermediate
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_features: If True, also return intermediate features
            
        Returns:
            logits: Class logits of shape (B, num_classes)
            features: (optional) Feature vector before classifier
        """
        features, intermediate = self.get_features(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features, intermediate
        return logits
    
    def get_model_size(self) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class StudentCNNSmall(StudentCNN):
    """Smaller variant of StudentCNN for mobile deployment."""
    
    def __init__(self, num_classes: int = 12, dropout_rate: float = 0.2):
        super().__init__(
            num_classes=num_classes,
            channels=[24, 48, 96, 192, 288],
            dropout_rate=dropout_rate,
            use_depthwise=True,
            use_se=True
        )


class StudentCNNTiny(StudentCNN):
    """Tiny variant of StudentCNN for extremely constrained devices."""
    
    def __init__(self, num_classes: int = 12, dropout_rate: float = 0.1):
        super().__init__(
            num_classes=num_classes,
            channels=[16, 32, 64, 128, 192],
            dropout_rate=dropout_rate,
            use_depthwise=True,
            use_se=False  # Disable SE for smaller size
        )


def create_student_model(
    model_type: str = "standard",
    num_classes: int = 12,
    **kwargs
) -> StudentCNN:
    """
    Factory function to create student models.
    
    Args:
        model_type: One of 'standard', 'small', 'tiny'
        num_classes: Number of output classes
        
    Returns:
        StudentCNN model instance
    """
    models = {
        "standard": StudentCNN,
        "small": StudentCNNSmall,
        "tiny": StudentCNNTiny
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model = models[model_type](num_classes=num_classes, **kwargs)
    
    # Log model size
    size_mb = model.get_model_size()
    logger.info(f"Created {model_type} student model: {size_mb:.2f} MB")
    
    return model


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = create_student_model("standard", num_classes=12)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test with features
    logits, features, intermediate = model(x, return_features=True)
    print(f"Feature shape: {features.shape}")
    print(f"Intermediate feature shapes: {[f.shape for f in intermediate]}")
    
    # Model size
    print(f"Model size: {model.get_model_size():.2f} MB")
