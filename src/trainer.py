"""
Knowledge Distillation Trainer
===============================
Multi-teacher knowledge distillation with comprehensive logging and metrics tracking.

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

try:
    import onnxruntime as ort  # type: ignore[import-unresolved]
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime-gpu")

logger = logging.getLogger(__name__)


class TeacherEnsemble:
    """
    Ensemble of ONNX teacher models for knowledge distillation.
    
    Features:
    - Load multiple ONNX models
    - Weighted ensemble predictions
    - Soft label generation with temperature scaling
    """
    
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        models_dir: str,
        teacher_num_classes: int = 11,
        device: str = "cuda"
    ):
        """
        Initialize teacher ensemble.
        
        Args:
            model_configs: List of model configurations with name, path, weight
            models_dir: Directory containing ONNX models
            teacher_num_classes: Number of classes teachers were trained on
            device: Device to use (cuda or cpu)
        """
        self.models_dir = Path(models_dir)
        self.teacher_num_classes = teacher_num_classes
        self.device = device
        self.sessions = {}
        self.weights = {}
        
        # Initialize ONNX sessions
        self._load_models(model_configs)
        
        logger.info(f"Loaded {len(self.sessions)} teacher models")
    
    def _load_models(self, model_configs: List[Dict[str, Any]]):
        """Load ONNX models."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is required for teacher ensemble")
        
        # Use CPU for ONNX inference (more reliable, GPU used for student training)
        # This avoids CUDA/cuDNN version compatibility issues with onnxruntime-gpu
        providers = ['CPUExecutionProvider']
        
        for config in model_configs:
            name = config['name']
            path = self.models_dir / config['path']
            weight = config.get('weight', 1.0)
            
            if not path.exists():
                logger.warning(f"Model not found: {path}")
                continue
            
            try:
                session = ort.InferenceSession(str(path), providers=providers)
                self.sessions[name] = session
                self.weights[name] = weight
                logger.info(f"  Loaded: {name} (weight={weight})")
            except Exception as e:
                logger.error(f"  Failed to load {name}: {e}")
    
    def get_soft_labels(
        self,
        images: torch.Tensor,
        temperature: float = 4.0,
        student_num_classes: int = 12
    ) -> torch.Tensor:
        """
        Generate soft labels from teacher ensemble.
        
        Args:
            images: Input images (B, C, H, W)
            temperature: Temperature for softening
            student_num_classes: Number of classes for student
            
        Returns:
            Weighted soft labels from all teachers
        """
        # Convert images to numpy for ONNX
        images_np = images.cpu().numpy()
        
        all_logits = []
        total_weight = 0
        
        for name, session in self.sessions.items():
            weight = self.weights[name]
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            try:
                outputs = session.run(None, {input_name: images_np})
                logits = torch.tensor(outputs[0], device=images.device)
                
                # Handle class mismatch: pad or truncate
                if logits.shape[1] != student_num_classes:
                    if logits.shape[1] < student_num_classes:
                        # Pad with zeros for new classes
                        padding = torch.zeros(
                            logits.shape[0], 
                            student_num_classes - logits.shape[1],
                            device=logits.device
                        )
                        logits = torch.cat([logits, padding], dim=1)
                    else:
                        # Truncate (shouldn't happen in our case)
                        logits = logits[:, :student_num_classes]
                
                all_logits.append(logits * weight)
                total_weight += weight
                
            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {e}")
                continue
        
        if not all_logits:
            # Return uniform distribution if no teachers available
            return torch.ones(images.shape[0], student_num_classes, device=images.device) / student_num_classes
        
        # Weighted average of logits
        ensemble_logits = sum(all_logits) / total_weight
        
        # Apply temperature scaling and softmax
        soft_labels = F.softmax(ensemble_logits / temperature, dim=1)
        
        return soft_labels


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Loss = alpha * KD_loss + beta * CE_loss
    
    Where:
    - KD_loss: KL divergence between student and teacher soft labels
    - CE_loss: Cross entropy with hard labels
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        temperature: float = 4.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_soft_labels: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined distillation loss.
        
        Args:
            student_logits: Raw logits from student
            teacher_soft_labels: Soft labels from teacher ensemble
            hard_labels: Ground truth labels
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary with individual loss values
        """
        # Soft label loss (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        kd_loss = self.kl_loss(student_soft, teacher_soft_labels) * (self.temperature ** 2)
        
        # Hard label loss (Cross entropy)
        ce_loss = self.ce_loss(student_logits, hard_labels)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + self.beta * ce_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }
        
        return total_loss, loss_components


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = defaultdict(list)
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_components: Dict[str, float]
    ):
        """Update metrics with batch results."""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        for key, value in loss_components.items():
            self.losses[key].append(value)
    
    def compute(self) -> Dict[str, Any]:
        """Compute all metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Overall accuracy
        accuracy = (predictions == targets).mean()
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx in range(self.num_classes):
            class_mask = targets == class_idx
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == class_idx).mean()
            else:
                class_acc = 0.0
            
            pred_mask = predictions == class_idx
            tp = ((predictions == class_idx) & (targets == class_idx)).sum()
            fp = ((predictions == class_idx) & (targets != class_idx)).sum()
            fn = ((predictions != class_idx) & (targets == class_idx)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[self.class_names[class_idx]] = {
                'accuracy': float(class_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(class_mask.sum())
            }
        
        # Confusion matrix
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for pred, target in zip(predictions, targets):
            confusion_matrix[target, pred] += 1
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in self.losses.items()}
        
        # Macro averages
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
        
        return {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'per_class': per_class_metrics,
            'confusion_matrix': confusion_matrix.tolist(),
            **avg_losses
        }


class KnowledgeDistillationTrainer:
    """
    Main trainer for knowledge distillation.
    
    Features:
    - Multi-teacher ensemble
    - Mixed precision training
    - Comprehensive logging
    - Checkpoint saving
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_ensemble: TeacherEnsemble,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        dataset_info: Dict[str, Any],
        output_dir: str
    ):
        self.student = student_model
        self.teachers = teacher_ensemble
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.dataset_info = dataset_info
        self.output_dir = Path(output_dir)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = self.student.to(self.device)
        
        # Setup loss
        self.criterion = DistillationLoss(
            alpha=config['distillation']['alpha'],
            beta=config['distillation']['beta'],
            temperature=config['distillation']['temperature']
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config['training'].get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Metrics tracking
        self.train_metrics = MetricsTracker(
            dataset_info['num_classes'],
            dataset_info['class_names']
        )
        self.val_metrics = MetricsTracker(
            dataset_info['num_classes'],
            dataset_info['class_names']
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        config = self.config['training']
        
        if config['optimizer'].lower() == 'adamw':
            return torch.optim.AdamW(
                self.student.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'].lower() == 'adam':
            return torch.optim.Adam(
                self.student.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        else:
            return torch.optim.SGD(
                self.student.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        config = self.config['training']
        
        if config['scheduler'].lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=1e-6
            )
        elif config['scheduler'].lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'training_{timestamp}.log'
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging to: {log_file}")
    
    def train_epoch(self) -> Dict[str, Any]:
        """Train for one epoch."""
        self.student.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get soft labels from teachers
            with torch.no_grad():
                soft_labels = self.teachers.get_soft_labels(
                    images,
                    temperature=self.config['distillation']['temperature'],
                    student_num_classes=self.dataset_info['num_classes']
                )
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.student(images)
                    loss, loss_components = self.criterion(logits, soft_labels, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.student(images)
                loss, loss_components = self.criterion(logits, soft_labels, labels)
                loss.backward()
                
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            predictions = logits.argmax(dim=1)
            self.train_metrics.update(predictions, labels, loss_components)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_components['total_loss']:.4f}",
                'acc': f"{(predictions == labels).float().mean():.4f}"
            })
        
        return self.train_metrics.compute()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """Validate the model."""
        self.student.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get soft labels from teachers
            soft_labels = self.teachers.get_soft_labels(
                images,
                temperature=self.config['distillation']['temperature'],
                student_num_classes=self.dataset_info['num_classes']
            )
            
            # Forward pass
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.student(images)
                    _, loss_components = self.criterion(logits, soft_labels, labels)
            else:
                logits = self.student(images)
                _, loss_components = self.criterion(logits, soft_labels, labels)
            
            # Update metrics
            predictions = logits.argmax(dim=1)
            self.val_metrics.update(predictions, labels, loss_components)
            
            pbar.set_postfix({
                'acc': f"{(predictions == labels).float().mean():.4f}"
            })
        
        return self.val_metrics.compute()
    
    def save_checkpoint(self, metrics: Dict[str, Any], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'metrics': metrics,
            'config': self.config,
            'dataset_info': self.dataset_info
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pt')
        
        # Save periodic checkpoint
        save_freq = self.config['logging'].get('save_frequency', 5)
        if (self.current_epoch + 1) % save_freq == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pt')
            logger.info(f"Saved best model with accuracy: {metrics['accuracy']:.4f}")
    
    def save_metrics(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]):
        """Save metrics to JSON file."""
        metrics_dir = self.output_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        epoch_metrics = {
            'epoch': self.current_epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train': train_metrics,
            'val': val_metrics
        }
        
        self.training_history.append(epoch_metrics)
        
        # Save complete history
        with open(metrics_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save latest metrics
        with open(metrics_dir / 'latest_metrics.json', 'w') as f:
            json.dump(epoch_metrics, f, indent=2)
    
    def check_early_stopping(self, val_accuracy: float) -> bool:
        """Check if training should stop early."""
        early_stop_config = self.config['training'].get('early_stopping', {})
        
        if not early_stop_config.get('enabled', False):
            return False
        
        min_delta = early_stop_config.get('min_delta', 0.001)
        patience = early_stop_config.get('patience', 15)
        
        if val_accuracy > self.best_val_accuracy + min_delta:
            self.best_val_accuracy = val_accuracy
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                return True
        
        return False
    
    def train(self):
        """Run full training loop."""
        epochs = self.config['training']['epochs']
        
        logger.info("="*60)
        logger.info("Starting Knowledge Distillation Training")
        logger.info("="*60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Training samples: {self.dataset_info['train_samples']}")
        logger.info(f"Validation samples: {self.dataset_info['val_samples']}")
        logger.info(f"Number of classes: {self.dataset_info['num_classes']}")
        logger.info("="*60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Check for best model
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
            
            # Save checkpoint and metrics
            self.save_checkpoint(val_metrics, is_best)
            self.save_metrics(train_metrics, val_metrics)
            
            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Best: {self.best_val_accuracy:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping
            if self.check_early_stopping(val_metrics['accuracy']):
                break
        
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info("="*60)
        
        # Save final summary
        self._save_training_summary(total_time)
        
        return self.best_val_accuracy
    
    def _save_training_summary(self, total_time: float):
        """Save training summary."""
        summary = {
            'total_time_seconds': total_time,
            'total_epochs': self.current_epoch + 1,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'dataset_info': self.dataset_info,
            'final_metrics': self.training_history[-1] if self.training_history else None
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Test components
    logging.basicConfig(level=logging.INFO)
    
    print("Knowledge Distillation Trainer module loaded successfully")
