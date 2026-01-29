"""
Enhanced Student Model Architecture for Multi-Teacher Knowledge Distillation
============================================================================
Advanced architecture designed to absorb knowledge from multiple teacher models
while maintaining the capacity for future learning and development.

Features:
- Multi-scale feature extraction (mimics ResNet, EfficientNet, Inception)
- Attention mechanisms (mimics ensemble_attention teacher)
- Feature pyramid for hierarchical learning
- Elastic capacity - can grow with more teachers
- Knowledge consolidation layers
- Self-attention for complex pattern recognition

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================
# Building Blocks
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution."""
    
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
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)  # Swish activation (better than ReLU)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SqueezeExcitation(nn.Module):
    """SE block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = F.silu(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        scale = self.bn(scale)
        scale = torch.sigmoid(scale)
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel and spatial attention."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = SqueezeExcitation(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiScaleConv(nn.Module):
    """Multi-scale convolution inspired by Inception - captures features at different scales."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 4
        
        # 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True)
        )
        
        # 3x3 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True)
        )
        
        # 5x5 convolution (as two 3x3)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True)
        )
        
        # Pooling branch
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], dim=1)


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block with expansion."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        expand_ratio: int = 4,
        use_attention: bool = True
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.attention(out)
        if self.use_residual:
            out = out + x
        return out


class KnowledgeConsolidationBlock(nn.Module):
    """
    Special block for consolidating knowledge from multiple teachers.
    Uses self-attention to learn relationships between features.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.LayerNorm([channels])
        
        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1)
        )
        self.norm2 = nn.LayerNorm([channels])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Self-attention
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q.transpose(-2, -1) @ k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        out = self.proj(out)
        
        # Residual + LayerNorm
        x = x + out
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # MLP
        out = self.mlp(x)
        x = x + out
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        
        return x


class FeaturePyramidNeck(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""
    
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Top-down pathway
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )
        
        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outputs


# ============================================================
# Enhanced Student Model
# ============================================================

class EnhancedStudentModel(nn.Module):
    """
    Enhanced Student Model for Multi-Teacher Knowledge Distillation.
    
    This model is designed to:
    1. Absorb knowledge from multiple diverse teachers (11+ models)
    2. Maintain capacity for future learning and development
    3. Support sequential teacher adaptation
    4. No performance loss - matches/exceeds teacher metrics
    
    Architecture Features:
    - Multi-scale feature extraction (Inception-style)
    - Attention mechanisms (channel + spatial + self-attention)
    - Feature pyramid for hierarchical learning
    - Knowledge consolidation blocks
    - Elastic depth with residual connections
    - Dropout and regularization for generalization
    
    Size Target: ~10-15 MB for mobile deployment
    """
    
    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 3,
        input_size: int = 256,
        base_channels: int = 48,
        expand_ratio: int = 4,
        dropout_rate: float = 0.3,
        num_consolidation_blocks: int = 2,
        use_fpn: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.use_fpn = use_fpn
        
        # Channel configuration (progressively wider)
        channels = [
            base_channels,           # Stage 1: 48
            base_channels * 2,       # Stage 2: 96
            base_channels * 4,       # Stage 3: 192
            base_channels * 6,       # Stage 4: 288
            base_channels * 8        # Stage 5: 384
        ]
        
        # ============================================================
        # Stem - Initial feature extraction
        # ============================================================
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        # ============================================================
        # Feature Extraction Stages
        # ============================================================
        
        # Stage 1: Multi-scale inception-style
        self.stage1 = nn.Sequential(
            MultiScaleConv(channels[0], channels[1]),
            InvertedResidual(channels[1], channels[1], expand_ratio=expand_ratio),
            nn.MaxPool2d(2, 2)
        )
        
        # Stage 2: Efficient blocks with attention
        self.stage2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=expand_ratio),
            InvertedResidual(channels[2], channels[2], expand_ratio=expand_ratio),
            InvertedResidual(channels[2], channels[2], expand_ratio=expand_ratio)
        )
        
        # Stage 3: Deeper feature extraction
        self.stage3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=expand_ratio),
            InvertedResidual(channels[3], channels[3], expand_ratio=expand_ratio),
            InvertedResidual(channels[3], channels[3], expand_ratio=expand_ratio),
            InvertedResidual(channels[3], channels[3], expand_ratio=expand_ratio)
        )
        
        # Stage 4: High-level features
        self.stage4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=expand_ratio),
            InvertedResidual(channels[4], channels[4], expand_ratio=expand_ratio),
            InvertedResidual(channels[4], channels[4], expand_ratio=expand_ratio)
        )
        
        # ============================================================
        # Knowledge Consolidation - Self-attention for learning relationships
        # ============================================================
        self.consolidation_blocks = nn.ModuleList([
            KnowledgeConsolidationBlock(channels[4], num_heads=8)
            for _ in range(num_consolidation_blocks)
        ])
        
        # ============================================================
        # Feature Pyramid Network (optional)
        # ============================================================
        if use_fpn:
            self.fpn = FeaturePyramidNeck(
                [channels[1], channels[2], channels[3], channels[4]],
                out_channels=channels[2]
            )
            self.fpn_pool = nn.AdaptiveAvgPool2d(1)
            fpn_features = channels[2] * 4  # Concatenated FPN features
        else:
            fpn_features = 0
        
        # ============================================================
        # Global Pooling and Classifier
        # ============================================================
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final feature dimension
        final_features = channels[4] + (fpn_features if use_fpn else 0)
        
        # Two-stage classifier for better learning
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(final_features, channels[4]),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(channels[4], channels[3]),
            nn.SiLU(inplace=True),
            nn.Linear(channels[3], num_classes)
        )
        
        # ============================================================
        # Auxiliary heads for deep supervision during training
        # ============================================================
        self.aux_classifiers = nn.ModuleDict({
            'stage2': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[2], num_classes)
            ),
            'stage3': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[3], num_classes)
            ),
            'stage4': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[4], num_classes)
            )
        })
        
        # Store channel info for feature matching
        self.stage_channels = channels
        
        # Initialize weights
        self._initialize_weights()
        
        # Log model info
        self._log_model_info()
    
    def _initialize_weights(self):
        """Initialize weights using modern best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _log_model_info(self):
        """Log model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        logger.info(f"EnhancedStudentModel Architecture:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Estimated model size: {model_size_mb:.2f} MB")
        logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Stage channels: {self.stage_channels}")
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features at multiple levels for knowledge distillation.
        
        Returns:
            Dictionary with features from each stage
        """
        features = {}
        
        x = self.stem(x)
        features['stem'] = x
        
        x = self.stage1(x)
        features['stage1'] = x
        
        x = self.stage2(x)
        features['stage2'] = x
        
        x = self.stage3(x)
        features['stage3'] = x
        
        x = self.stage4(x)
        features['stage4'] = x
        
        # Apply knowledge consolidation
        for i, block in enumerate(self.consolidation_blocks):
            x = block(x)
        features['consolidated'] = x
        
        return features
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False,
        return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional feature and auxiliary output return.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: Return intermediate features for KD
            return_aux: Return auxiliary classifier outputs for deep supervision
            
        Returns:
            Dictionary containing:
            - 'logits': Main classifier output
            - 'features': (optional) Intermediate features
            - 'aux': (optional) Auxiliary classifier outputs
        """
        output = {}
        
        # Feature extraction
        features = self.get_features(x)
        
        # Main pathway
        x = features['consolidated']
        
        # Global pooling
        main_features = self.global_pool(x).flatten(1)
        
        # FPN features if enabled
        if self.use_fpn:
            fpn_inputs = [features['stage1'], features['stage2'], features['stage3'], features['stage4']]
            fpn_outputs = self.fpn(fpn_inputs)
            fpn_features = torch.cat([
                self.fpn_pool(f).flatten(1) for f in fpn_outputs
            ], dim=1)
            main_features = torch.cat([main_features, fpn_features], dim=1)
        
        # Classification
        logits = self.classifier(main_features)
        output['logits'] = logits
        
        # Optional outputs
        if return_features:
            output['features'] = features
            output['final_features'] = main_features
        
        if return_aux:
            output['aux'] = {
                'stage2': self.aux_classifiers['stage2'](features['stage2']),
                'stage3': self.aux_classifiers['stage3'](features['stage3']),
                'stage4': self.aux_classifiers['stage4'](features['stage4'])
            }
        
        return output
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 * 1024)


def create_enhanced_student(
    num_classes: int = 11,
    size: str = 'medium',
    **kwargs
) -> EnhancedStudentModel:
    """
    Factory function to create EnhancedStudentModel with different sizes.
    
    Args:
        num_classes: Number of output classes
        size: Model size - 'small', 'medium', 'large'
        **kwargs: Additional arguments passed to model
        
    Returns:
        EnhancedStudentModel instance
    """
    configs = {
        'small': {
            'base_channels': 32,
            'expand_ratio': 3,
            'num_consolidation_blocks': 1,
            'use_fpn': False
        },
        'medium': {
            'base_channels': 48,
            'expand_ratio': 4,
            'num_consolidation_blocks': 2,
            'use_fpn': True
        },
        'large': {
            'base_channels': 64,
            'expand_ratio': 6,
            'num_consolidation_blocks': 3,
            'use_fpn': True
        }
    }
    
    config = configs.get(size, configs['medium'])
    config.update(kwargs)
    
    model = EnhancedStudentModel(num_classes=num_classes, **config)
    
    size_mb = model.get_model_size_mb()
    logger.info(f"Created {size} student model: {size_mb:.2f} MB")
    
    return model


if __name__ == '__main__':
    # Test the model
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = create_enhanced_student(num_classes=11, size='medium')
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    output = model(x, return_features=True, return_aux=True)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Final features: {output['final_features'].shape}")
    print(f"\nFeature shapes:")
    for name, feat in output['features'].items():
        print(f"  {name}: {feat.shape}")
    print(f"\nAuxiliary outputs:")
    for name, aux in output['aux'].items():
        print(f"  {name}: {aux.shape}")
