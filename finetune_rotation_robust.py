"""
Rotation-Robust Fine-tuning Script
===================================
Fine-tune the existing trained student model with rotation augmentation
using the SAME sequential training approach as the original distillation.

This script:
1. Loads the already trained student model (96.25% accuracy)
2. Fine-tunes it with rotation augmentation
3. Uses sequential teacher learning like original training
4. Uses same hyperparameters (T=4.0, alpha=0.6, beta=0.4, EWC)

Author: Intelli-PEST Backend Team
Date: 2026-01-01
"""

import os
import sys
import json
import logging
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import random
import copy

# Windows multiprocessing fix - must be before torch imports
if sys.platform == 'win32':
    multiprocessing.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from enhanced_student_model import EnhancedStudentModel, create_enhanced_student  # pyright: ignore
from teacher_loader import MultiFormatTeacherEnsemble  # pyright: ignore

# Import shared training utilities for Ghost-alignment
# Note: training_utils provides standardized kd_loss with proper class mismatch handling
sys.path.insert(0, str(Path(__file__).parent.parent / 'Intelli_PEST-Backend' / 'black_ops_training'))
try:
    from training_utils import (  # pyright: ignore
        kd_loss as ghost_kd_loss,
        extract_logits,
        BatchConfig,
        MAX_PHYSICAL_BATCH_SIZE,
        MIN_KEY_LOAD_RATIO,
        MIN_VALID_TEACHERS,
        check_recovery_trigger,
        RecoveryState,
        UnifiedLogger,
        load_checkpoint_flexible,
        PreflightChecker,
        CheckSeverity,
        PreflightCheckResult,
        ExitCode,
        get_teacher_lr_multiplier,
        PerTeacherConfig,
    )
    TRAINING_UTILS_AVAILABLE = True
except ImportError:
    TRAINING_UTILS_AVAILABLE = False
    MIN_KEY_LOAD_RATIO = 0.80
    MIN_VALID_TEACHERS = 8

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('exported_models', exist_ok=True)

# Setup logging
log_file = f'logs/rotation_finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# Rotation Augmentation
# ============================================================

class RotationAugmentation:
    """
    Apply cardinal rotation augmentation (0°, 90°, 180°, 270°) to simulate EXIF rotations.
    All four cardinal rotations are equally likely when augmentation is applied.
    Returns the image and whether a non-zero rotation was applied.
    """
    
    def __init__(self, prob: float = 0.8):
        """
        Args:
            prob: Probability of applying any rotation (including 0°).
                  When applied, each of 0°, 90°, 180°, 270° has 25% chance.
        """
        self.prob = prob
        # All four cardinal rotations for EXIF simulation
        self.cardinal_angles = [0, 90, 180, 270]
    
    def __call__(self, img: Image.Image) -> Tuple[Image.Image, bool]:
        """
        Returns:
            img: Transformed image
            is_rotated: True if rotation != 0° was applied
        """
        is_rotated = False
        if random.random() < self.prob:
            # Equal probability for all 4 cardinal rotations
            angle = random.choice(self.cardinal_angles)
            if angle != 0:
                # Use BILINEAR for quality, expand=True to prevent cropping
                img = img.rotate(-angle, expand=True, resample=Image.BILINEAR)
                is_rotated = True
        return img, is_rotated


# ============================================================
# Dataset with Rotation Augmentation
# ============================================================

class RotationRobustDataset(Dataset):
    """Dataset with rotation augmentation for fine-tuning."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        image_size: int = 256,  # Match original training!
        rotation_prob: float = 0.8
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        # Rotation augmentation (only for training)
        self.rotation_aug = RotationAugmentation(prob=rotation_prob) if split == "train" else None
        
        # Transforms matching original training
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Discover classes
        self.class_names = sorted([
            d.name for d in self.root_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Load and split samples
        self.samples = self._load_and_split_samples(train_ratio, seed)
        
        logger.info(f"Dataset [{split}]: {len(self.samples)} samples, {self.num_classes} classes")
    
    def _load_and_split_samples(self, train_ratio: float, seed: int) -> List[Tuple[str, int]]:
        all_samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    all_samples.append((str(img_path), class_idx))
        
        # Split by class
        np.random.seed(int(seed))  # Ensure seed is int
        class_samples = {}
        for path, label in all_samples:
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append((path, label))
        
        train_samples, val_samples = [], []
        for label, samples in class_samples.items():
            np.random.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            train_samples.extend(samples[:split_idx])
            val_samples.extend(samples[split_idx:])
        
        result = train_samples if self.split == "train" else val_samples
        np.random.shuffle(result)
        return result
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        """
        Returns:
            img: Transformed image tensor
            label: Class label
            is_rotated: Whether non-zero rotation was applied
        """
        img_path, label = self.samples[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        # Apply rotation augmentation BEFORE other transforms
        is_rotated = False
        if self.rotation_aug:
            img, is_rotated = self.rotation_aug(img)
        
        img = self.transform(img)
        return img, label, is_rotated


# ============================================================
# EWC Loss (from original training)
# ============================================================

class EWCLoss(nn.Module):
    """Elastic Weight Consolidation to prevent forgetting.
    
    Added max_loss clamping to prevent EWC from dominating the training loss.
    """
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 100.0, max_loss: float = 1.0):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.max_loss = max_loss  # Clamp to prevent explosion
        self.params = {}
        self.fisher = {}
        self._initialized = False
        self._initialize(model)
    
    def _initialize(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
    
    def update_fisher(self, model: nn.Module, dataloader: DataLoader, device: str = 'cuda'):
        """Update Fisher information after learning from a teacher."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        
        num_batches = min(50, len(dataloader))  # Limit for speed
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Handle both 2-tuple and 3-tuple returns from dataset
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            model.zero_grad()
            output = model(images)
            logits = output['logits'] if isinstance(output, dict) else output
            
            pred_labels = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, pred_labels)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize and update
        for name in fisher:
            fisher[name] /= num_batches
            if name in self.fisher:
                self.fisher[name] = 0.5 * self.fisher[name] + 0.5 * fisher[name]
            else:
                self.fisher[name] = fisher[name]
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
        
        self._initialized = True
        logger.info(f"  EWC Fisher updated - {len(self.fisher)} parameters tracked")
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        if not self._initialized or len(self.fisher) == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        loss = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                loss += (self.fisher[name] * (param - self.params[name]) ** 2).sum()
        
        raw_loss = self.ewc_lambda * loss
        # Clamp to prevent EWC from dominating training
        clamped_loss = torch.clamp(raw_loss, max=self.max_loss)
        return clamped_loss


# ============================================================
# Sequential Fine-tuning Trainer
# ============================================================

class SequentialFineTuner:
    """
    Fine-tune student model with rotation augmentation using sequential learning.
    Same approach as original training.
    """
    
    def __init__(
        self,
        student: nn.Module,
        teachers: MultiFormatTeacherEnsemble,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.student = student.to(device)
        self.teachers = teachers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training params - 25 epochs per teacher with early stopping
        self.epochs_per_teacher = config.get('epochs_per_teacher', 25)
        self.self_refinement_epochs = config.get('self_refinement_epochs', 25)
        # Dual-metric early stopping
        self.upright_patience = config.get('upright_patience', 5)  # Patience for upright accuracy
        self.rotation_patience = config.get('rotation_patience', 10)  # Patience for rotation accuracy
        self.temperature = config.get('temperature', 4.0)
        self.alpha = config.get('alpha', 0.6)  # Soft label weight (upright only)
        self.beta = config.get('beta', 0.4)    # Hard label weight (upright only)
        self.lr = config.get('learning_rate', 0.0005)  # Lower LR for fine-tuning
        self.min_lr = config.get('min_learning_rate', 1e-6)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.use_ewc = config.get('use_ewc', True)
        self.ewc_lambda = config.get('ewc_lambda', 100.0)  # Reduced from 500 to prevent explosion
        self.max_ewc_loss = config.get('max_ewc_loss', 1.0)  # Clamp EWC loss to prevent dominating
        
        # Setup training
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        self.scaler = GradScaler('cuda')
        
        # EWC - with clamping to prevent explosion
        if self.use_ewc:
            self.ewc_loss = EWCLoss(self.student, self.ewc_lambda, max_loss=self.max_ewc_loss)
        else:
            self.ewc_loss = None
        self.ewc_active = False  # Only activate after first teacher
        
        # Tracking - dual metrics (upright and rotation accuracy)
        self.best_upright_accuracy = 0.0
        self.best_rotation_accuracy = 0.0
        self.best_state = None
        self.history = {'per_teacher': {}, 'final': {}}
        
        logger.info(f"Sequential Fine-tuner initialized")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Alpha (soft): {self.alpha}, Beta (hard): {self.beta}")
        logger.info(f"  Learning rate: {self.lr}")
        logger.info(f"  Epochs per teacher: {self.epochs_per_teacher}")
        logger.info(f"  Self-refinement epochs: {self.self_refinement_epochs}")
        logger.info(f"  DUAL-METRIC EARLY STOPPING:")
        logger.info(f"    - Upright patience: {self.upright_patience} epochs")
        logger.info(f"    - Rotation patience: {self.rotation_patience} epochs")
        logger.info(f"    - BOTH conditions must be met to trigger early stop")
        logger.info(f"  EWC enabled: {self.use_ewc}")
        if self.use_ewc:
            logger.info(f"    - EWC lambda: {self.ewc_lambda}")
            logger.info(f"    - EWC max loss clamp: {self.max_ewc_loss}")
        logger.info(f"  NOTE: KD loss for upright images only, CE loss for rotated images")
    
    def train(self, skip_teachers: int = 0) -> Dict[str, Any]:
        """Run sequential fine-tuning with self-refinement phase.
        
        Args:
            skip_teachers: Number of teachers to skip (for resuming from checkpoint)
        """
        logger.info("="*70)
        logger.info("STARTING SEQUENTIAL FINE-TUNING WITH ROTATION AUGMENTATION")
        logger.info("="*70)
        
        teacher_names = self.teachers.get_teacher_names()
        total_teachers = len(teacher_names)
        
        if skip_teachers > 0:
            logger.info(f"RESUMING: Skipping first {skip_teachers} teachers")
            # Activate EWC since we're resuming after at least one teacher
            if self.use_ewc:
                self.ewc_active = True
                logger.info(f"  EWC activated (resumed mode)")
        
        # ============================================================
        # PHASE 1: Sequential learning from each teacher
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: SEQUENTIAL TEACHER LEARNING")
        logger.info("="*70)
        
        for idx, teacher_name in enumerate(teacher_names):
            # Skip teachers if resuming
            if idx < skip_teachers:
                logger.info(f"Skipping teacher {idx+1}/{total_teachers}: {teacher_name}")
                continue
            
            # Step 1.6: Use PerTeacherConfig for standardized teacher training configuration
            if TRAINING_UTILS_AVAILABLE:
                # Get teacher accuracy if available (default to 90%)
                teacher_accuracy = self.teacher_ensemble.get_teacher_accuracy(teacher_name) if hasattr(self.teacher_ensemble, 'get_teacher_accuracy') else 90.0
                
                teacher_config = PerTeacherConfig.from_teacher(
                    teacher_name=teacher_name,
                    teacher_index=idx + 1,
                    total_teachers=total_teachers,
                    base_lr=self.lr,
                    teacher_accuracy=teacher_accuracy,
                    warmup_epochs=2,  # 2 epochs warmup per teacher
                    main_epochs=self.config['epochs_per_teacher'] - 2,
                    num_classes=11
                )
                teacher_config.log_config(logger.info)
                
                # Apply teacher-specific learning rate
                current_lr = teacher_config.effective_lr * (0.9 ** idx)  # Additional decay per teacher
                current_lr = max(current_lr, self.min_lr)
            else:
                logger.info(f"\n{'='*70}")
                logger.info(f"TEACHER {idx+1}/{total_teachers}: {teacher_name}")
                logger.info(f"{'='*70}")
                
                # Fallback: More aggressive learning rate decay for stability
                # Use exponential decay: 0.9 per teacher (was 0.95)
                decay_factor = 0.9
                current_lr = self.lr * (decay_factor ** idx)
                current_lr = max(current_lr, self.min_lr)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            logger.info(f"  Final Learning rate: {current_lr:.6f}")
            
            # Train with this teacher
            teacher_metrics = self._train_with_teacher(teacher_name, idx)
            self.history['per_teacher'][teacher_name] = teacher_metrics
            
            # Check if teacher caused collapse
            if teacher_metrics.get('collapsed', False):
                logger.warning(f"  Teacher {teacher_name} caused collapse - skipping to next teacher")
                continue
            
            # Update EWC after each teacher (but only activate after first)
            if self.use_ewc:
                logger.info(f"Updating EWC for {teacher_name}...")
                self.ewc_loss.update_fisher(self.student, self.train_loader, self.device)
                if not self.ewc_active:
                    self.ewc_active = True
                    logger.info(f"  EWC activated - will apply penalty from next teacher")
                    logger.info(f"  EWC lambda: {self.ewc_lambda}, max_loss: {self.max_ewc_loss}")
            
            # Save checkpoint
            self._save_checkpoint(f"finetune_after_{teacher_name}")
        
        # ============================================================
        # PHASE 2: Self-refinement with 100% rotation
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: SELF-REFINEMENT WITH 100% ROTATION")
        logger.info("="*70)
        logger.info("Training on ground truth only with 100% rotation augmentation")
        
        # Create dataset with 100% rotation probability
        train_dataset_100_rot = RotationRobustDataset(
            root_dir=self.train_loader.dataset.root_dir,
            split="train",
            image_size=self.config['image_size'],
            rotation_prob=1.0  # 100% rotation for self-refinement
        )
        
        train_loader_100_rot = DataLoader(
            train_dataset_100_rot,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        logger.info(f"Self-refinement dataset: {len(train_dataset_100_rot)} samples with 100% rotation")
        
        self_refinement_metrics = self._self_refinement_phase(train_loader_100_rot)
        self.history['self_refinement'] = self_refinement_metrics
        
        # Save final model
        self._save_final_model()
        
        return self._generate_report()
    
    def _train_with_teacher(self, teacher_name: str, teacher_idx: int) -> Dict[str, Any]:
        """Train with a single teacher with DUAL-METRIC early stopping.
        
        Early stopping requires BOTH conditions to be met:
        1. Upright accuracy no improvement for upright_patience epochs
        2. Rotation accuracy no improvement for rotation_patience epochs
        
        Also includes:
        - Warmup for first 2 epochs of each teacher
        - Collapse detection and recovery
        - Per-epoch checkpoint saving
        """
        metrics = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_upright_acc': [], 'val_rotation_acc': [],
            'best_upright_acc': 0.0, 'best_rotation_acc': 0.0
        }
        
        # Dual-metric early stopping tracking
        best_teacher_upright = 0.0
        best_teacher_rotation = 0.0
        epochs_no_upright_improve = 0
        epochs_no_rotation_improve = 0
        
        # Collapse detection - track previous epoch accuracy
        prev_train_acc = None
        pre_teacher_state = copy.deepcopy(self.student.state_dict())  # Save state before teacher
        
        # Get base learning rate for this teacher (already set by caller)
        base_lr = self.optimizer.param_groups[0]['lr']
        warmup_epochs = 2  # Warmup for first 2 epochs
        
        for epoch in range(self.epochs_per_teacher):
            # Learning rate warmup for first few epochs of each teacher
            if epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"    [Warmup] epoch {epoch+1}/{warmup_epochs}, LR: {warmup_lr:.6f}")
            else:
                # Restore base LR after warmup
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr
            # Training
            train_loss, train_acc = self._train_epoch(teacher_name)
            
            # Dual validation: upright + rotation
            val_loss, upright_acc = self._validate()  # Upright validation
            rotation_acc = self._validate_rotation()   # Rotation validation (avg of 90°, 180°, 270°)
            
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_loss'].append(val_loss)
            metrics['val_upright_acc'].append(upright_acc)
            metrics['val_rotation_acc'].append(rotation_acc)
            
            # Track upright accuracy improvement
            if upright_acc > best_teacher_upright:
                best_teacher_upright = upright_acc
                epochs_no_upright_improve = 0
            else:
                epochs_no_upright_improve += 1
            
            # Track rotation accuracy improvement
            if rotation_acc > best_teacher_rotation:
                best_teacher_rotation = rotation_acc
                epochs_no_rotation_improve = 0
            else:
                epochs_no_rotation_improve += 1
            
            metrics['best_upright_acc'] = max(metrics['best_upright_acc'], upright_acc)
            metrics['best_rotation_acc'] = max(metrics['best_rotation_acc'], rotation_acc)
            
            # Track global best (save model when upright improves)
            if upright_acc > self.best_upright_accuracy:
                self.best_upright_accuracy = upright_acc
                self.best_state = copy.deepcopy(self.student.state_dict())
            if rotation_acc > self.best_rotation_accuracy:
                self.best_rotation_accuracy = rotation_acc
            
            logger.info(
                f"  Epoch {epoch+1}/{self.epochs_per_teacher} - "
                f"Train: {train_acc:.2f}% | "
                f"Upright: {upright_acc:.2f}% (best: {best_teacher_upright:.2f}%, stall: {epochs_no_upright_improve}/{self.upright_patience}) | "
                f"Rotation: {rotation_acc:.2f}% (best: {best_teacher_rotation:.2f}%, stall: {epochs_no_rotation_improve}/{self.rotation_patience})"
            )
            
            # ============================================================
            # Collapse detection: accuracy drops > 30% from previous epoch
            # ============================================================
            if prev_train_acc is not None and (prev_train_acc - train_acc) > 30:
                logger.warning(f"  ⚠️ COLLAPSE DETECTED! Training accuracy dropped from {prev_train_acc:.2f}% to {train_acc:.2f}%")
                logger.warning(f"  Restoring pre-teacher state and skipping {teacher_name}")
                self.student.load_state_dict(pre_teacher_state)
                metrics['collapsed'] = True
                metrics['collapse_epoch'] = epoch + 1
                return metrics
            
            prev_train_acc = train_acc
            
            # Save per-epoch checkpoint for recovery
            if epoch > 0 and (epoch + 1) % 5 == 0:  # Every 5 epochs
                self._save_checkpoint(f"finetune_{teacher_name}_epoch{epoch+1}")
            
            # DUAL-METRIC Early stopping: BOTH conditions must be met
            upright_stalled = epochs_no_upright_improve >= self.upright_patience
            rotation_stalled = epochs_no_rotation_improve >= self.rotation_patience
            
            if upright_stalled and rotation_stalled:
                logger.info(f"  ★ DUAL-METRIC early stopping triggered at epoch {epoch+1}")
                logger.info(f"    Upright stalled for {epochs_no_upright_improve} epochs (patience={self.upright_patience})")
                logger.info(f"    Rotation stalled for {epochs_no_rotation_improve} epochs (patience={self.rotation_patience})")
                break
            elif upright_stalled:
                logger.info(f"    [Upright stalled] Continuing - rotation still improving")
            elif rotation_stalled:
                logger.info(f"    [Rotation stalled] Continuing - upright still improving")
        
        return metrics
    
    def _train_epoch(self, teacher_name: str) -> Tuple[float, float]:
        """Train one epoch with a specific teacher.
        
        Loss strategy:
        - Upright images (not rotated): KD loss (α×soft + β×hard) - learn from teacher
        - Rotated images: CE loss only - teachers would give wrong labels
        """
        self.student.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_upright = 0
        num_rotated = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training with {teacher_name}", leave=False)
        
        for images, labels, is_rotated in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            # Convert is_rotated to tensor and move to device
            is_rotated = torch.tensor(is_rotated, dtype=torch.bool, device=self.device)
            
            self.optimizer.zero_grad()
            
            with autocast('cuda'):
                # Student forward - model returns dict with 'logits' key
                output = self.student(images)
                student_logits = output['logits'] if isinstance(output, dict) else output
                
                # Get indices for upright and rotated samples
                upright_idx = torch.where(~is_rotated)[0]
                rotated_idx = torch.where(is_rotated)[0]
                
                loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                
                # Loss for UPRIGHT images: KD + CE (learn from teacher)
                if len(upright_idx) > 0:
                    upright_logits = student_logits[upright_idx]
                    upright_labels = labels[upright_idx]
                    upright_images = images[upright_idx]
                    
                    # Get teacher soft labels for upright images only
                    with torch.no_grad():
                        teacher_logits = self.teachers.get_teacher_predictions(upright_images, teacher_name)
                        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
                    
                    # KD loss with proper class mismatch handling (Ghost-aligned)
                    # Uses slicing to handle student having more classes than teacher
                    if TRAINING_UTILS_AVAILABLE:
                        kd_loss_val = ghost_kd_loss(upright_logits, teacher_soft, self.temperature)
                    else:
                        # Fallback: slice student logits to match teacher class count
                        num_teacher_classes = teacher_soft.shape[1]
                        num_student_classes = upright_logits.shape[1]
                        if num_student_classes > num_teacher_classes:
                            upright_logits_shared = upright_logits[:, :num_teacher_classes]
                        else:
                            upright_logits_shared = upright_logits
                        student_soft = F.log_softmax(upright_logits_shared / self.temperature, dim=-1)
                        kd_loss_val = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
                    
                    ce_loss_upright = F.cross_entropy(upright_logits, upright_labels)
                    
                    loss_upright = self.alpha * kd_loss_val + self.beta * ce_loss_upright
                    loss = loss + loss_upright * (len(upright_idx) / images.size(0))
                    num_upright += len(upright_idx)
                
                # Loss for ROTATED images: CE only (ground truth)
                if len(rotated_idx) > 0:
                    rotated_logits = student_logits[rotated_idx]
                    rotated_labels = labels[rotated_idx]
                    
                    ce_loss_rotated = F.cross_entropy(rotated_logits, rotated_labels)
                    loss = loss + ce_loss_rotated * (len(rotated_idx) / images.size(0))
                    num_rotated += len(rotated_idx)
                
                # Add EWC penalty (only after first teacher completes)
                ewc_penalty = torch.tensor(0.0, device=self.device)
                if self.use_ewc and self.ewc_loss is not None and self.ewc_active:
                    ewc_penalty = self.ewc_loss(self.student)
                    loss = loss + ewc_penalty
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Calculate gradient norm before clipping for monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            postfix = {
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'up/rot': f'{num_upright}/{num_rotated}'
            }
            if self.ewc_active:
                postfix['ewc'] = f'{ewc_penalty.item():.4f}'
            pbar.set_postfix(postfix)
        
        logger.info(f"    Upright samples: {num_upright}, Rotated samples: {num_rotated}")
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def _validate(self) -> Tuple[float, float]:
        """Validate the model on UPRIGHT images."""
        self.student.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both 2-tuple and 3-tuple returns from dataset
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                output = self.student(images)
                logits = output['logits'] if isinstance(output, dict) else output
                loss = F.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def _validate_rotation(self) -> float:
        """Validate the model on ROTATED images (90°, 180°, 270°).
        
        Returns average accuracy across the three rotations.
        This tests rotation robustness separately from upright accuracy.
        """
        self.student.eval()
        
        rotation_angles = [90, 180, 270]  # Option A: only rotated, not 0°
        accuracies = []
        
        with torch.no_grad():
            for angle in rotation_angles:
                correct = 0
                total = 0
                
                for batch in self.val_loader:
                    if len(batch) == 3:
                        images, labels, _ = batch
                    else:
                        images, labels = batch
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Apply rotation
                    # k=1 for 90°, k=2 for 180°, k=3 for 270°
                    k = angle // 90
                    rotated_images = torch.rot90(images, k=k, dims=[2, 3])
                    
                    output = self.student(rotated_images)
                    logits = output['logits'] if isinstance(output, dict) else output
                    
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                acc = 100. * correct / total
                accuracies.append(acc)
        
        avg_rotation_acc = sum(accuracies) / len(accuracies)
        return avg_rotation_acc
    
    def _self_refinement_phase(self, train_loader_100_rotation: DataLoader) -> Dict[str, Any]:
        """
        Self-refinement phase: Train student on its own predictions with 100% rotation.
        
        Since teachers weren't trained on rotation, in this phase we only use CE loss
        with the student's own knowledge (no teacher guidance).
        
        Uses DUAL-METRIC early stopping (both upright and rotation must stall).
        """
        logger.info("")
        logger.info("="*70)
        logger.info("SELF-REFINEMENT PHASE")
        logger.info("Rotation augmentation: 100%")
        logger.info(f"Epochs: {self.self_refinement_epochs}")
        logger.info(f"Dual-metric early stopping: upright_patience={self.upright_patience}, rotation_patience={self.rotation_patience}")
        logger.info("="*70)
        
        phase_history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_upright_acc': [], 'val_rotation_acc': []
        }
        
        # Dual-metric tracking
        best_upright = self.best_upright_accuracy
        best_rotation = self.best_rotation_accuracy
        epochs_no_upright_improve = 0
        epochs_no_rotation_improve = 0
        
        for epoch in range(self.self_refinement_epochs):
            # Training
            self.student.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader_100_rotation, desc=f"Self-Refinement Epoch {epoch+1}/{self.self_refinement_epochs}")
            
            for batch in pbar:
                # Get data (ignore rotation flag - we know it's 100% rotated)
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    output = self.student(images)
                    logits = output['logits'] if isinstance(output, dict) else output
                    # Only CE loss in self-refinement (no teacher)
                    loss = F.cross_entropy(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            train_loss = total_loss / len(train_loader_100_rotation)
            train_acc = 100. * correct / total
            
            # Dual validation
            val_loss, upright_acc = self._validate()
            rotation_acc = self._validate_rotation()
            
            phase_history['train_loss'].append(train_loss)
            phase_history['train_acc'].append(train_acc)
            phase_history['val_loss'].append(val_loss)
            phase_history['val_upright_acc'].append(upright_acc)
            phase_history['val_rotation_acc'].append(rotation_acc)
            
            # Track upright improvement
            if upright_acc > best_upright:
                best_upright = upright_acc
                self.best_upright_accuracy = upright_acc
                self.best_state = copy.deepcopy(self.student.state_dict())
                epochs_no_upright_improve = 0
                logger.info(f"  ★ New best upright accuracy: {upright_acc:.2f}%")
            else:
                epochs_no_upright_improve += 1
            
            # Track rotation improvement
            if rotation_acc > best_rotation:
                best_rotation = rotation_acc
                self.best_rotation_accuracy = rotation_acc
                epochs_no_rotation_improve = 0
                logger.info(f"  ★ New best rotation accuracy: {rotation_acc:.2f}%")
            else:
                epochs_no_rotation_improve += 1
            
            logger.info(
                f"Self-Refinement Epoch {epoch+1}/{self.self_refinement_epochs}: "
                f"Train={train_acc:.2f}% | "
                f"Upright={upright_acc:.2f}% (best: {best_upright:.2f}%, stall: {epochs_no_upright_improve}/{self.upright_patience}) | "
                f"Rotation={rotation_acc:.2f}% (best: {best_rotation:.2f}%, stall: {epochs_no_rotation_improve}/{self.rotation_patience})"
            )
            
            # DUAL-METRIC early stopping
            upright_stalled = epochs_no_upright_improve >= self.upright_patience
            rotation_stalled = epochs_no_rotation_improve >= self.rotation_patience
            
            if upright_stalled and rotation_stalled:
                logger.info(f"  ★ DUAL-METRIC early stopping triggered at epoch {epoch+1}")
                logger.info(f"    Upright stalled for {epochs_no_upright_improve} epochs")
                logger.info(f"    Rotation stalled for {epochs_no_rotation_improve} epochs")
                break
            elif upright_stalled:
                logger.info(f"    [Upright stalled] Continuing - rotation still improving")
            elif rotation_stalled:
                logger.info(f"    [Rotation stalled] Continuing - upright still improving")
        
        return {
            'best_upright_accuracy': best_upright,
            'best_rotation_accuracy': best_rotation,
            'history': phase_history
        }
    
    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        path = Path('checkpoints') / f'{name}.pt'
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_upright_accuracy': self.best_upright_accuracy,
            'best_rotation_accuracy': self.best_rotation_accuracy,
            'history': self.history,
        }, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _save_final_model(self):
        """Save the final fine-tuned model."""
        # Restore best state
        if self.best_state:
            self.student.load_state_dict(self.best_state)
        
        # Save PyTorch model
        output_path = Path('exported_models') / 'student_model_rotation_robust.pt'
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'best_upright_accuracy': self.best_upright_accuracy,
            'best_rotation_accuracy': self.best_rotation_accuracy,
            'training_config': self.config,
            'timestamp': datetime.now().isoformat(),
        }, output_path)
        logger.info(f"Saved final model: {output_path}")
        
        # Export to ONNX
        self._export_onnx()
    
    def _export_onnx(self):
        """Export model to ONNX format."""
        self.student.eval()
        
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        output_path = Path('exported_models') / 'student_model_rotation_robust.onnx'
        
        torch.onnx.export(
            self.student,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=13,
            do_constant_folding=True,
        )
        logger.info(f"Exported ONNX model: {output_path}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate training report."""
        report = {
            'best_upright_accuracy': self.best_upright_accuracy,
            'best_rotation_accuracy': self.best_rotation_accuracy,
            'config': self.config,
            'history': self.history,
            'timestamp': datetime.now().isoformat(),
        }
        
        report_path = Path('exported_models') / 'rotation_finetune_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved report: {report_path}")
        return report


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune student model with rotation augmentation')
    parser.add_argument('--epochs-per-teacher', type=int, default=25, help='Epochs per teacher (default: 25)')
    parser.add_argument('--self-refinement-epochs', type=int, default=25, help='Self-refinement epochs with 100%% rotation (default: 25)')
    parser.add_argument('--upright-patience', type=int, default=5, help='Early stopping patience for upright accuracy (default: 5)')
    parser.add_argument('--rotation-patience', type=int, default=10, help='Early stopping patience for rotation accuracy (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=12, help='Number of data loading workers (default: 12)')
    parser.add_argument('--rotation-prob', type=float, default=0.8, help='Rotation augmentation probability (default: 0.8)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (e.g., checkpoints/finetune_after_alexnet.pt)')
    parser.add_argument('--skip-teachers', type=int, default=0, help='Skip first N teachers (use with --resume)')
    parser.add_argument('--ewc-lambda', type=float, default=100.0, help='EWC lambda (default: 100.0, reduced from 500)')
    parser.add_argument('--max-ewc-loss', type=float, default=1.0, help='Max EWC loss clamp (default: 1.0)')
    parser.add_argument('--no-ewc', action='store_true', help='Disable EWC entirely')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration (matching original training with enhancements)
    config = {
        'epochs_per_teacher': args.epochs_per_teacher,  # 25 epochs per teacher
        'self_refinement_epochs': args.self_refinement_epochs,  # 25 epochs self-refinement with 100% rotation
        'upright_patience': args.upright_patience,  # 5 epochs patience for upright
        'rotation_patience': args.rotation_patience,  # 10 epochs patience for rotation
        'temperature': 4.0,  # Same as original
        'alpha': 0.6,        # Same as original (for upright images only)
        'beta': 0.4,         # Same as original (for upright images only)
        'learning_rate': args.lr,
        'min_learning_rate': 1e-6,
        'weight_decay': 0.01,
        'use_ewc': not args.no_ewc,  # Can disable EWC
        'ewc_lambda': args.ewc_lambda,  # Reduced default from 500 to 100
        'max_ewc_loss': args.max_ewc_loss,  # Clamp EWC loss
        'image_size': 256,    # Same as original (teacher models expect 256x256)
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,  # Multiprocessing workers (12)
        'rotation_prob': args.rotation_prob,
        'resume_checkpoint': args.resume,
        'skip_teachers': args.skip_teachers,
    }
    
    # Load config for teacher models
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # ============================================================
    # STEP 1.5: PRE-FLIGHT CHECKS
    # ============================================================
    if TRAINING_UTILS_AVAILABLE:
        logger.info("="*70)
        logger.info("RUNNING PRE-FLIGHT CHECKS (Step 1.5)")
        logger.info("="*70)
        
        # Get teacher paths
        teacher_config = yaml_config.get('teachers', {})
        teacher_dir = Path(teacher_config.get('pytorch_dir', 'D:/Intelli_PEST-Backend/teacher_models/pytorch'))
        teacher_paths = list(teacher_dir.glob("**/*.pt")) if teacher_dir.exists() else []
        
        # Get dataset path
        data_config = yaml_config.get('data', {})
        dataset_path = Path(data_config.get('train_dir', 'D:/IMAGE DATASET'))
        
        # Get model path
        model_path = Path('student_model_final.pth')
        if config['resume_checkpoint']:
            model_path = Path(config['resume_checkpoint'])
        
        preflight_checker = PreflightChecker(
            config={
                'output_dir': str(Path('checkpoints')),
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'epochs': config['epochs_per_teacher'] * 10,  # Rough estimate
                'temperature': config['temperature'],
                'mixup_alpha': 0.2,
                'cutmix_alpha': 1.0,
                'rotation_augmentation': True,
            },
            log_fn=logger.info,
            teacher_paths=teacher_paths,
            dataset_path=dataset_path,
            model_path=model_path
        )
        
        preflight_results = preflight_checker.run_all_checks()
        
        # Write preflight report
        try:
            unified_logger = UnifiedLogger(Path('logs'))
            unified_logger.write_preflight_report(preflight_results)
            logger.info(f"Pre-flight report saved to: {unified_logger.preflight_report}")
        except Exception as e:
            logger.warning(f"Could not save preflight report: {e}")
        
        # Abort on critical failures
        if preflight_checker.should_abort(preflight_results):
            logger.error("CRITICAL PRE-FLIGHT CHECK(S) FAILED - Aborting")
            sys.exit(preflight_checker.get_exit_code(preflight_results))
        
        logger.info("Pre-flight checks passed")
    
    # ============================================================
    # Load existing trained student model
    # ============================================================
    logger.info("="*70)
    logger.info("LOADING TRAINED STUDENT MODEL")
    logger.info("="*70)
    
    # Check if resuming from a finetune checkpoint
    if config['resume_checkpoint']:
        resume_path = Path(config['resume_checkpoint'])
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        
        logger.info(f"RESUMING from checkpoint: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device)
        
        logger.info(f"  Checkpoint keys: {resume_checkpoint.keys()}")
        logger.info(f"  Best upright accuracy: {resume_checkpoint.get('best_upright_accuracy', 'N/A'):.2f}%")
        logger.info(f"  Best rotation accuracy: {resume_checkpoint.get('best_rotation_accuracy', 'N/A'):.2f}%")
        
        # Create student model
        student_config = yaml_config.get('student', {})
        student = create_enhanced_student(
            num_classes=student_config.get('num_classes', 11),
            size=student_config.get('size', 'medium'),
            base_channels=student_config.get('base_channels', 48),
            expand_ratio=student_config.get('expand_ratio', 4),
            dropout_rate=student_config.get('dropout_rate', 0.3),
            num_consolidation_blocks=student_config.get('num_consolidation_blocks', 2),
            use_fpn=student_config.get('use_fpn', True)
        )
        
        # Step 1.4: Use flexible checkpoint loading for resume
        if TRAINING_UTILS_AVAILABLE:
            load_result = load_checkpoint_flexible(
                model=student,
                checkpoint_path=resume_path,
                device=device,
                min_load_ratio=MIN_KEY_LOAD_RATIO,
                log_details=True
            )
            if not load_result.success:
                raise RuntimeError(f"Resume checkpoint load failed: {load_result.error}")
            logger.info(f"Loaded from resume checkpoint: method={load_result.method}, "
                       f"keys={load_result.keys_loaded}/{load_result.keys_total} ({load_result.load_ratio:.1%})")
        else:
            student.load_state_dict(resume_checkpoint['model_state_dict'])
        
        student = student.to(device)
        
        logger.info(f"Loaded model from finetune checkpoint")
    else:
        # Load the original trained model checkpoint
        checkpoint_path = Path('student_model_final.pth')
        
        # Create student model with same architecture as original training
        student_config = yaml_config.get('student', {})
        student = create_enhanced_student(
            num_classes=student_config.get('num_classes', 11),
            size=student_config.get('size', 'medium'),
            base_channels=student_config.get('base_channels', 48),
            expand_ratio=student_config.get('expand_ratio', 4),
            dropout_rate=student_config.get('dropout_rate', 0.3),
            num_consolidation_blocks=student_config.get('num_consolidation_blocks', 2),
            use_fpn=student_config.get('use_fpn', True)
        )
        
        # Step 1.4: Use flexible checkpoint loading
        if TRAINING_UTILS_AVAILABLE:
            load_result = load_checkpoint_flexible(
                model=student,
                checkpoint_path=checkpoint_path,
                device=device,
                min_load_ratio=MIN_KEY_LOAD_RATIO,
                log_details=True
            )
            if not load_result.success:
                raise RuntimeError(f"Checkpoint load failed: {load_result.error}")
            logger.info(f"Loaded checkpoint: method={load_result.method}, "
                       f"keys={load_result.keys_loaded}/{load_result.keys_total} ({load_result.load_ratio:.1%})")
            
            # Read metadata from checkpoint for logging
            checkpoint = torch.load(checkpoint_path, map_location=device)
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            if 'best_accuracy' in checkpoint:
                logger.info(f"Original accuracy: {checkpoint['best_accuracy']*100:.2f}%")
            if 'teachers_used' in checkpoint:
                logger.info(f"Teachers used: {checkpoint['teachers_used']}")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            logger.info(f"Original accuracy: {checkpoint['best_accuracy']*100:.2f}%")
            logger.info(f"Teachers used: {checkpoint['teachers_used']}")
            student.load_state_dict(checkpoint['model_state_dict'])
        
        student = student.to(device)
    
    logger.info(f"Student model loaded successfully")
    logger.info(f"Model size: {student.get_model_size_mb():.2f} MB")
    
    # ============================================================
    # Create datasets with rotation augmentation
    # ============================================================
    logger.info("="*70)
    logger.info("CREATING DATASETS WITH ROTATION AUGMENTATION")
    logger.info("="*70)
    
    data_dir = yaml_config['dataset']['path']
    
    train_dataset = RotationRobustDataset(
        root_dir=data_dir,
        split="train",
        image_size=config['image_size'],
        rotation_prob=config['rotation_prob']
    )
    
    val_dataset = RotationRobustDataset(
        root_dir=data_dir,
        split="val",
        image_size=config['image_size'],
        rotation_prob=0.0  # No rotation for validation
    )
    
    # Configure batch size with hard limit enforcement (Step 1.3)
    if TRAINING_UTILS_AVAILABLE:
        batch_config = BatchConfig.from_requested(
            requested_batch_size=config['batch_size'],
            max_physical=MAX_PHYSICAL_BATCH_SIZE,
            enable_adaptive_probe=True
        )
        batch_config.log_config(logger.info)
        physical_batch_size = batch_config.physical_batch_size
        config['accumulation_steps'] = batch_config.accumulation_steps
    else:
        physical_batch_size = min(config['batch_size'], 16)  # Hard limit fallback
        config['accumulation_steps'] = max(1, config['batch_size'] // physical_batch_size)
        logger.info(f"Batch Configuration (fallback):")
        logger.info(f"  Physical: {physical_batch_size}, Accumulation: {config['accumulation_steps']}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,  # Use enforced physical limit
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=physical_batch_size,  # Use enforced physical limit
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    logger.info(f"Using {config['num_workers']} workers for data loading (multiprocessing)")
    
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # ============================================================
    # Load teacher ensemble
    # ============================================================
    logger.info("="*70)
    logger.info("LOADING TEACHER ENSEMBLE")
    logger.info("="*70)
    
    teachers = MultiFormatTeacherEnsemble(
        model_configs=yaml_config['teachers']['models'],
        models_dir=yaml_config['teachers']['models_dir'],
        num_classes=yaml_config['teachers']['teacher_num_classes'],
        device=device
    )
    
    logger.info(f"Loaded {teachers.get_teacher_count()} teachers")
    
    # ============================================================
    # Fine-tune with rotation augmentation
    # ============================================================
    logger.info("="*70)
    logger.info("STARTING FINE-TUNING")
    logger.info("="*70)
    
    trainer = SequentialFineTuner(
        student=student,
        teachers=teachers,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Pass skip_teachers for resume functionality
    results = trainer.train(skip_teachers=config.get('skip_teachers', 0))
    
    logger.info("="*70)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("="*70)
    logger.info(f"Final upright accuracy: {results['best_upright_accuracy']:.2f}%")
    logger.info(f"Final rotation accuracy: {results['best_rotation_accuracy']:.2f}%")


if __name__ == '__main__':
    main()
