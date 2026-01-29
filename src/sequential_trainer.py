"""
Sequential Knowledge Distillation Trainer
==========================================
Train student model by learning from teachers one-by-one,
allowing proper adaptation and knowledge consolidation.

Key Features:
1. Sequential teacher learning - each teacher adapts the student
2. Knowledge consolidation between teachers
3. Elastic Weight Consolidation (EWC) to prevent forgetting
4. Comprehensive metrics tracking
5. Deep supervision with auxiliary losses
6. Multi-format teacher support

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # type: ignore
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import json
import logging
from datetime import datetime
from tqdm import tqdm
import copy

from .teacher_loader import MultiFormatTeacherEnsemble
from .enhanced_student_model import EnhancedStudentModel, create_enhanced_student

logger = logging.getLogger(__name__)


# ============================================================
# Loss Functions
# ============================================================

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    - Hard label loss (CrossEntropy)
    - Soft label loss (KL Divergence)
    - Feature matching loss (optional)
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,  # Weight for soft labels
        beta: float = 0.5,   # Weight for hard labels
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        self.hard_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_soft_labels: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Raw logits from student (B, C)
            teacher_soft_labels: Soft labels from teacher (B, C)
            hard_labels: Ground truth labels (B,)
            
        Returns:
            Dict with total_loss and component losses
        """
        # Hard label loss
        loss_hard = self.hard_loss(student_logits, hard_labels)
        
        # Soft label loss (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss_soft = self.kl_loss(student_soft, teacher_soft_labels) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * loss_soft + self.beta * loss_hard
        
        return {
            'total': total_loss,
            'hard': loss_hard,
            'soft': loss_soft
        }


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC) loss to prevent catastrophic forgetting
    when learning from multiple teachers sequentially.
    """
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.params = {}
        self.fisher = {}
        self._initialize(model)
    
    def _initialize(self, model: nn.Module):
        """Store initial parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
    
    def update_fisher(self, model: nn.Module, dataloader: DataLoader, device: str = 'cuda'):
        """
        Update Fisher information after learning from a teacher.
        This estimates parameter importance for preventing forgetting.
        """
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        
        for batch in dataloader:
            images = batch[0].to(device)
            
            model.zero_grad()
            output = model(images)
            logits = output['logits'] if isinstance(output, dict) else output
            
            # Use empirical Fisher (log-likelihood gradient squared)
            labels = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize
        n_samples = len(dataloader)
        for name in fisher:
            fisher[name] /= n_samples
        
        # Update stored Fisher and parameters
        for name in fisher:
            if name in self.fisher:
                self.fisher[name] = 0.5 * self.fisher[name] + 0.5 * fisher[name]
            else:
                self.fisher[name] = fisher[name]
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty."""
        loss = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                loss += (self.fisher[name] * (param - self.params[name]) ** 2).sum()
        return self.ewc_lambda * loss


# ============================================================
# Sequential Distillation Trainer
# ============================================================

class SequentialDistillationTrainer:
    """
    Trainer for sequential knowledge distillation from multiple teachers.
    
    Training Process:
    1. Initialize student model
    2. For each teacher (in order):
       a. Train student to match teacher outputs
       b. Update EWC to prevent forgetting
       c. Log and save metrics
    3. Final ensemble refinement (optional)
    4. Export final model
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        student_model: EnhancedStudentModel,
        teachers: MultiFormatTeacherEnsemble,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        device: str = 'cuda'
    ):
        """
        Initialize sequential distillation trainer.
        
        Args:
            config: Training configuration
            student_model: Student model to train
            teachers: Multi-format teacher ensemble
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Output directory for checkpoints/logs
            device: Training device
        """
        self.config = config
        self.student = student_model.to(device)
        self.teachers = teachers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Training settings
        self.epochs_per_teacher = config.get('epochs_per_teacher', 10)
        self.final_ensemble_epochs = config.get('final_ensemble_epochs', 20)
        self.temperature = config.get('temperature', 4.0)
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.5)
        self.use_ewc = config.get('use_ewc', True)
        self.ewc_lambda = config.get('ewc_lambda', 1000.0)
        self.use_aux_loss = config.get('use_aux_loss', True)
        self.aux_weight = config.get('aux_weight', 0.3)
        
        # Learning rate settings
        self.lr = config.get('learning_rate', 0.001)
        self.min_lr = config.get('min_learning_rate', 1e-6)
        self.weight_decay = config.get('weight_decay', 0.01)
        
        # Initialize components
        self._setup_training()
        
        # Metrics tracking
        self.metrics_history = {
            'per_teacher': {},
            'final': {},
            'teacher_performance': {}
        }
        
        # Create output directories
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Sequential Distillation Trainer initialized")
        logger.info(f"  - Teachers: {teachers.get_teacher_count()}")
        logger.info(f"  - Epochs per teacher: {self.epochs_per_teacher}")
        logger.info(f"  - Final ensemble epochs: {self.final_ensemble_epochs}")
        logger.info(f"  - EWC enabled: {self.use_ewc}")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, losses."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Distillation loss
        self.distill_loss = DistillationLoss(
            temperature=self.temperature,
            alpha=self.alpha,
            beta=self.beta
        )
        
        # EWC loss (if enabled)
        if self.use_ewc:
            self.ewc_loss = EWCLoss(self.student, self.ewc_lambda)
        else:
            self.ewc_loss = None
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda')
        
        # Best model tracking
        self.best_accuracy = 0.0
        self.best_state = None
    
    def train(self) -> Dict[str, Any]:
        """
        Run full sequential training pipeline.
        
        Returns:
            Dictionary with training metrics and results
        """
        logger.info("="*70)
        logger.info("STARTING SEQUENTIAL KNOWLEDGE DISTILLATION")
        logger.info("="*70)
        
        teacher_names = self.teachers.get_teacher_names()
        total_teachers = len(teacher_names)
        
        # Phase 1: Sequential teacher learning
        for idx, teacher_name in enumerate(teacher_names):
            logger.info(f"\n{'='*70}")
            logger.info(f"PHASE {idx+1}/{total_teachers}: Learning from {teacher_name}")
            logger.info(f"{'='*70}")
            
            # Adjust learning rate based on progress
            current_lr = self.lr * (0.9 ** idx)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(current_lr, self.min_lr)
            
            # Train with this teacher
            teacher_metrics = self._train_with_teacher(teacher_name, idx)
            self.metrics_history['per_teacher'][teacher_name] = teacher_metrics
            
            # Update EWC after learning from each teacher
            if self.use_ewc:
                logger.info(f"Updating EWC for {teacher_name}...")
                self.ewc_loss.update_fisher(self.student, self.train_loader, self.device)
            
            # Save checkpoint after each teacher
            self._save_checkpoint(f"after_{teacher_name}")
        
        # Phase 2: Final ensemble refinement
        if self.final_ensemble_epochs > 0:
            logger.info(f"\n{'='*70}")
            logger.info("PHASE FINAL: Ensemble Refinement")
            logger.info(f"{'='*70}")
            
            final_metrics = self._train_with_ensemble()
            self.metrics_history['final'] = final_metrics
        
        # Save final model
        self._save_final_model()
        
        # Generate final report
        report = self._generate_report()
        
        return report
    
    def _train_with_teacher(self, teacher_name: str, teacher_idx: int) -> Dict[str, Any]:
        """Train student with a single teacher."""
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_acc': 0.0
        }
        
        for epoch in range(self.epochs_per_teacher):
            # Training
            train_loss, train_acc = self._train_epoch_single_teacher(teacher_name)
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self._validate()
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            # Update best
            if val_acc > metrics['best_acc']:
                metrics['best_acc'] = val_acc
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_state = copy.deepcopy(self.student.state_dict())
            
            logger.info(
                f"[{teacher_name}] Epoch {epoch+1}/{self.epochs_per_teacher} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
            )
        
        return metrics
    
    def _train_epoch_single_teacher(self, teacher_name: str) -> Tuple[float, float]:
        """Train one epoch with a single teacher."""
        self.student.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training [{teacher_name}]", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast('cuda'):
                # Get teacher predictions
                teacher_logits = self.teachers.get_teacher_predictions(images, teacher_name)
                teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
                
                # Student forward
                output = self.student(images, return_aux=self.use_aux_loss)
                student_logits = output['logits']
                
                # Distillation loss
                losses = self.distill_loss(student_logits, teacher_soft, labels)
                loss = losses['total']
                
                # Auxiliary loss (deep supervision)
                if self.use_aux_loss and 'aux' in output:
                    for aux_name, aux_logits in output['aux'].items():
                        aux_loss = self.distill_loss(aux_logits, teacher_soft, labels)
                        loss = loss + self.aux_weight * aux_loss['total']
                
                # EWC loss
                if self.ewc_loss is not None:
                    loss = loss + self.ewc_loss(self.student)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    def _train_with_ensemble(self) -> Dict[str, Any]:
        """Final training phase with all teachers as ensemble."""
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_acc': 0.0
        }
        
        # Lower learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr * 10
        
        for epoch in range(self.final_ensemble_epochs):
            # Training with full ensemble
            train_loss, train_acc = self._train_epoch_ensemble()
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self._validate()
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            if val_acc > metrics['best_acc']:
                metrics['best_acc'] = val_acc
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_state = copy.deepcopy(self.student.state_dict())
            
            logger.info(
                f"[Ensemble] Epoch {epoch+1}/{self.final_ensemble_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
            )
        
        return metrics
    
    def _train_epoch_ensemble(self) -> Tuple[float, float]:
        """Train one epoch with full teacher ensemble."""
        self.student.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training [Ensemble]", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast('cuda'):
                # Get ensemble soft labels
                teacher_soft = self.teachers.get_soft_labels(images, self.temperature)
                
                # Student forward
                output = self.student(images, return_aux=self.use_aux_loss)
                student_logits = output['logits']
                
                # Distillation loss
                losses = self.distill_loss(student_logits, teacher_soft, labels)
                loss = losses['total']
                
                # Auxiliary loss
                if self.use_aux_loss and 'aux' in output:
                    for aux_logits in output['aux'].values():
                        aux_loss = self.distill_loss(aux_logits, teacher_soft, labels)
                        loss = loss + self.aux_weight * aux_loss['total']
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    def _validate(self) -> Tuple[float, float]:
        """Validate student model."""
        self.student.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                output = self.student(images)
                logits = output['logits'] if isinstance(output, dict) else output
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), correct / total
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.checkpoints_dir / f"checkpoint_{name}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _save_final_model(self):
        """Save final trained model."""
        # Load best state
        if self.best_state is not None:
            self.student.load_state_dict(self.best_state)
        
        # Save complete model
        model_path = self.output_dir / 'student_model_final.pth'
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'model_config': {
                'num_classes': self.student.num_classes,
                'input_size': self.student.input_size
            },
            'best_accuracy': self.best_accuracy,
            'training_config': self.config,
            'teachers_used': self.teachers.get_teacher_names(),
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Saved final model: {model_path}")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.2%}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate training report."""
        report = {
            'summary': {
                'best_accuracy': self.best_accuracy,
                'total_teachers': self.teachers.get_teacher_count(),
                'epochs_per_teacher': self.epochs_per_teacher,
                'final_ensemble_epochs': self.final_ensemble_epochs
            },
            'per_teacher_results': {},
            'final_results': self.metrics_history.get('final', {}),
            'model_info': {
                'size_mb': self.student.get_model_size_mb(),
                'num_classes': self.student.num_classes
            }
        }
        
        # Add per-teacher results
        for teacher_name, metrics in self.metrics_history.get('per_teacher', {}).items():
            report['per_teacher_results'][teacher_name] = {
                'best_accuracy': metrics.get('best_acc', 0),
                'final_train_acc': metrics['train_acc'][-1] if metrics.get('train_acc') else 0,
                'final_val_acc': metrics['val_acc'][-1] if metrics.get('val_acc') else 0
            }
        
        # Save report
        report_path = self.output_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved: {report_path}")
        
        return report


# ============================================================
# Factory Function
# ============================================================

def create_sequential_trainer(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    device: str = 'cuda'
) -> SequentialDistillationTrainer:
    """
    Factory function to create a configured sequential trainer.
    
    Args:
        config: Full configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Output directory
        device: Training device
        
    Returns:
        Configured SequentialDistillationTrainer
    """
    # Create student model
    student_config = config.get('student', {})
    student = create_enhanced_student(
        num_classes=student_config.get('num_classes', 11),
        size=student_config.get('size', 'medium')
    )
    
    # Create teacher ensemble
    teacher_config = config.get('teachers', {})
    teachers = MultiFormatTeacherEnsemble(
        model_configs=teacher_config.get('models', []),
        models_dir=teacher_config.get('models_dir', ''),
        num_classes=teacher_config.get('teacher_num_classes', 11),
        device=device
    )
    
    # Create trainer
    training_config = config.get('training', {})
    trainer = SequentialDistillationTrainer(
        config=training_config,
        student_model=student,
        teachers=teachers,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device
    )
    
    return trainer
