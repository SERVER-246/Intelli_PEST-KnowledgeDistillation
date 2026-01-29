"""
Multi-Format Teacher Model Loader
=================================
Unified loader supporting multiple model formats:
- PyTorch (.pt, .pth)
- ONNX (.onnx)
- TensorFlow Lite (.tflite)

Each format is loaded with optimal inference settings.

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available - ONNX models will be skipped")

try:
    import tensorflow as tf  # type: ignore[import-unresolved]
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    logger.warning("tensorflow not available - TFLite models will be skipped")


# ============================================================
# Abstract Base Class for Teacher Models
# ============================================================

class TeacherModel(ABC):
    """Abstract base class for teacher models of any format."""
    
    def __init__(self, name: str, path: Path, weight: float = 1.0, num_classes: int = 11):
        self.name = name
        self.path = path
        self.weight = weight
        self.num_classes = num_classes
        self.format = self._detect_format()
    
    def _detect_format(self) -> str:
        suffix = self.path.suffix.lower()
        format_map = {
            '.pt': 'pytorch',
            '.pth': 'pytorch',
            '.onnx': 'onnx',
            '.tflite': 'tflite'
        }
        return format_map.get(suffix, 'unknown')
    
    @abstractmethod
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Run inference on images.
        
        Args:
            images: Input images as numpy array (B, C, H, W) or (B, H, W, C)
            
        Returns:
            Logits as numpy array (B, num_classes)
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is successfully loaded."""
        pass


# ============================================================
# PyTorch Teacher Model (.pt, .pth)
# ============================================================

class PyTorchTeacher(TeacherModel):
    """Teacher model loaded from PyTorch checkpoint."""
    
    def __init__(self, name: str, path: Path, weight: float = 1.0, 
                 num_classes: int = 11, device: str = 'cuda'):
        super().__init__(name, path, weight, num_classes)
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load PyTorch model from checkpoint."""
        try:
            checkpoint = torch.load(self.path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Need model architecture - try to infer or use generic
                    logger.warning(f"{self.name}: state_dict only - need architecture")
                    self.model = None
                elif 'model_state_dict' in checkpoint:
                    logger.warning(f"{self.name}: model_state_dict only - need architecture")
                    self.model = None
                else:
                    # Might be the model directly
                    self.model = checkpoint
            else:
                # Assume it's the model object
                self.model = checkpoint
            
            if self.model is not None:
                self.model.eval()
                self.model.to(self.device)
                logger.info(f"  Loaded PyTorch: {self.name} (weight={self.weight})")
        except Exception as e:
            logger.error(f"  Failed to load PyTorch {self.name}: {e}")
            self.model = None
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run inference with PyTorch model."""
        if self.model is None:
            return np.zeros((images.shape[0], self.num_classes))
        
        with torch.no_grad():
            x = torch.from_numpy(images).float().to(self.device)
            outputs = self.model(x)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            return outputs.cpu().numpy()
    
    def is_loaded(self) -> bool:
        return self.model is not None


# ============================================================
# ONNX Teacher Model (.onnx)
# ============================================================

class ONNXTeacher(TeacherModel):
    """Teacher model loaded from ONNX format."""
    
    def __init__(self, name: str, path: Path, weight: float = 1.0, 
                 num_classes: int = 11, use_gpu: bool = False):
        super().__init__(name, path, weight, num_classes)
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model(use_gpu)
    
    def _load_model(self, use_gpu: bool):
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            logger.warning(f"  Skipping ONNX {self.name}: onnxruntime not available")
            return
        
        try:
            # Use CPU for reliability (GPU can have CUDA version issues)
            providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(str(self.path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"  Loaded ONNX: {self.name} (weight={self.weight})")
        except Exception as e:
            logger.error(f"  Failed to load ONNX {self.name}: {e}")
            self.session = None
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run inference with ONNX model."""
        if self.session is None:
            return np.zeros((images.shape[0], self.num_classes))
        
        try:
            outputs = self.session.run(None, {self.input_name: images.astype(np.float32)})
            return outputs[0]
        except Exception as e:
            logger.error(f"ONNX inference error for {self.name}: {e}")
            return np.zeros((images.shape[0], self.num_classes))
    
    def is_loaded(self) -> bool:
        return self.session is not None


# ============================================================
# TensorFlow Lite Teacher Model (.tflite)
# ============================================================

class TFLiteTeacher(TeacherModel):
    """Teacher model loaded from TensorFlow Lite format."""
    
    def __init__(self, name: str, path: Path, weight: float = 1.0, 
                 num_classes: int = 11):
        super().__init__(name, path, weight, num_classes)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model."""
        if not TFLITE_AVAILABLE:
            logger.warning(f"  Skipping TFLite {self.name}: tensorflow not available")
            return
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"  Loaded TFLite: {self.name} (weight={self.weight})")
        except Exception as e:
            logger.error(f"  Failed to load TFLite {self.name}: {e}")
            self.interpreter = None
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run inference with TFLite model."""
        if self.interpreter is None:
            return np.zeros((images.shape[0], self.num_classes))
        
        try:
            batch_size = images.shape[0]
            results = []
            
            # Check expected input format
            input_shape = self.input_details[0]['shape']
            
            # TFLite often expects NHWC format
            if len(input_shape) == 4 and input_shape[3] == 3:
                # Convert from NCHW to NHWC
                images = np.transpose(images, (0, 2, 3, 1))
            
            # TFLite processes one image at a time typically
            for i in range(batch_size):
                img = np.expand_dims(images[i], axis=0).astype(np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], img)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                results.append(output[0])
            
            return np.array(results)
        except Exception as e:
            logger.error(f"TFLite inference error for {self.name}: {e}")
            return np.zeros((images.shape[0], self.num_classes))
    
    def is_loaded(self) -> bool:
        return self.interpreter is not None


# ============================================================
# Multi-Format Teacher Ensemble
# ============================================================

class MultiFormatTeacherEnsemble:
    """
    Ensemble of teacher models supporting multiple formats.
    
    Features:
    - Load models from .pt, .pth, .onnx, .tflite
    - Weighted soft label generation
    - Sequential teacher adaptation support
    - Format-agnostic interface
    """
    
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        models_dir: Union[str, Path],
        num_classes: int = 11,
        device: str = 'cuda'
    ):
        """
        Initialize multi-format teacher ensemble.
        
        Args:
            model_configs: List of dicts with 'name', 'path', 'weight', optional 'format'
            models_dir: Base directory for model files
            num_classes: Number of output classes
            device: Device for PyTorch models
        """
        self.models_dir = Path(models_dir)
        self.num_classes = num_classes
        self.device = device
        self.teachers: Dict[str, TeacherModel] = {}
        self.teacher_order: List[str] = []  # For sequential training
        
        self._load_all_models(model_configs)
    
    def _load_all_models(self, model_configs: List[Dict[str, Any]]):
        """Load all teacher models from various formats."""
        logger.info(f"Loading teacher models from: {self.models_dir}")
        
        for config in model_configs:
            name = config['name']
            path = self.models_dir / config['path']
            weight = config.get('weight', 1.0)
            
            if not path.exists():
                # Try to find model in different formats
                path = self._find_model_file(name)
                if path is None:
                    logger.warning(f"  Model not found: {name}")
                    continue
            
            teacher = self._load_single_model(name, path, weight)
            if teacher is not None and teacher.is_loaded():
                self.teachers[name] = teacher
                self.teacher_order.append(name)
        
        logger.info(f"Successfully loaded {len(self.teachers)} teacher models")
    
    def _find_model_file(self, name: str) -> Optional[Path]:
        """Try to find model file in different formats."""
        extensions = ['.onnx', '.pt', '.pth', '.tflite']
        
        for ext in extensions:
            path = self.models_dir / f"{name}{ext}"
            if path.exists():
                return path
        
        # Try subdirectories
        for ext in extensions:
            for subdir in ['onnx_models', 'pytorch_models', 'tflite_models']:
                path = self.models_dir.parent / subdir / f"{name}{ext}"
                if path.exists():
                    return path
        
        return None
    
    def _load_single_model(self, name: str, path: Path, weight: float) -> Optional[TeacherModel]:
        """Load a single model based on its format."""
        suffix = path.suffix.lower()
        
        try:
            if suffix in ['.pt', '.pth']:
                return PyTorchTeacher(name, path, weight, self.num_classes, self.device)
            elif suffix == '.onnx':
                return ONNXTeacher(name, path, weight, self.num_classes)
            elif suffix == '.tflite':
                return TFLiteTeacher(name, path, weight, self.num_classes)
            else:
                logger.warning(f"  Unknown format for {name}: {suffix}")
                return None
        except Exception as e:
            logger.error(f"  Error loading {name}: {e}")
            return None
    
    def get_soft_labels(
        self,
        images: torch.Tensor,
        temperature: float = 4.0,
        teacher_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Generate soft labels from teacher ensemble.
        
        Args:
            images: Input images (B, C, H, W)
            temperature: Softmax temperature
            teacher_names: Optional list of specific teachers to use
            
        Returns:
            Weighted soft labels (B, num_classes)
        """
        images_np = images.cpu().numpy()
        
        teachers_to_use = teacher_names if teacher_names else list(self.teachers.keys())
        
        all_logits = []
        total_weight = 0
        
        for name in teachers_to_use:
            if name not in self.teachers:
                continue
            
            teacher = self.teachers[name]
            weight = teacher.weight
            
            try:
                logits = teacher.predict(images_np)
                all_logits.append(logits * weight)
                total_weight += weight
            except Exception as e:
                logger.warning(f"Inference failed for {name}: {e}")
        
        if not all_logits:
            return torch.zeros(images.size(0), self.num_classes, device=images.device)
        
        # Weighted average
        ensemble_logits = np.sum(all_logits, axis=0) / total_weight
        
        # Convert to soft labels with temperature
        soft_labels = torch.from_numpy(ensemble_logits).float()
        soft_labels = torch.softmax(soft_labels / temperature, dim=-1)
        
        return soft_labels.to(images.device)
    
    def get_teacher_predictions(
        self,
        images: torch.Tensor,
        teacher_name: str
    ) -> torch.Tensor:
        """
        Get predictions from a specific teacher.
        
        Args:
            images: Input images (B, C, H, W)
            teacher_name: Name of the teacher model
            
        Returns:
            Logits from the teacher (B, num_classes)
        """
        if teacher_name not in self.teachers:
            raise ValueError(f"Teacher not found: {teacher_name}")
        
        images_np = images.cpu().numpy()
        logits = self.teachers[teacher_name].predict(images_np)
        
        return torch.from_numpy(logits).float().to(images.device)
    
    def get_teacher_names(self) -> List[str]:
        """Get list of loaded teacher names in order."""
        return self.teacher_order.copy()
    
    def get_teacher_count(self) -> int:
        """Get number of loaded teachers."""
        return len(self.teachers)
    
    def get_teacher_weights(self) -> Dict[str, float]:
        """Get weights for all teachers."""
        return {name: t.weight for name, t in self.teachers.items()}
    
    def get_teacher_formats(self) -> Dict[str, str]:
        """Get format of each teacher."""
        return {name: t.format for name, t in self.teachers.items()}


# ============================================================
# Testing
# ============================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    test_configs = [
        {'name': 'mobilenet_v2', 'path': 'mobilenet_v2.onnx', 'weight': 1.0},
        {'name': 'super_ensemble', 'path': 'super_ensemble.onnx', 'weight': 2.0}
    ]
    
    # Test ensemble
    models_dir = Path(r"D:\Intelli_PEST-Backend\tflite_models_compatible\onnx_models")
    
    if models_dir.exists():
        ensemble = MultiFormatTeacherEnsemble(
            test_configs, 
            models_dir,
            num_classes=11
        )
        
        print(f"\nLoaded teachers: {ensemble.get_teacher_names()}")
        print(f"Teacher formats: {ensemble.get_teacher_formats()}")
        
        # Test inference
        test_images = torch.randn(2, 3, 256, 256)
        soft_labels = ensemble.get_soft_labels(test_images, temperature=3.0)
        print(f"Soft labels shape: {soft_labels.shape}")
    else:
        print(f"Models directory not found: {models_dir}")
