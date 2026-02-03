"""
Sequential Knowledge Distillation Training Script
=================================================
Main entry point for training the enhanced student model
using sequential learning from all 11 teacher models.

Features:
1. Sequential teacher adaptation (one teacher at a time)
2. Elastic Weight Consolidation to prevent forgetting
3. Multi-format teacher support (.pt, .pth, .onnx, .tflite)
4. Comprehensive metrics tracking
5. Model export to all formats

Usage:
    python train_sequential.py --config configs/config.yaml

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import create_dataloaders
from src.enhanced_student_model import create_enhanced_student
from src.evaluator import ModelEvaluator
from src.exporter import ModelExporter
from src.sequential_trainer import SequentialDistillationTrainer
from src.teacher_loader import MultiFormatTeacherEnsemble

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dirs(base_dir: Path) -> dict:
    """Create output directories."""
    dirs = {
        'checkpoints': base_dir / 'checkpoints',
        'logs': base_dir / 'logs',
        'exported_models': base_dir / 'exported_models',
        'metrics': base_dir / 'metrics'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def main():
    parser = argparse.ArgumentParser(description='Sequential Knowledge Distillation Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup output directories
    output_dir = Path(args.output_dir)
    dirs = setup_output_dirs(output_dir)

    # Setup file logging
    log_file = dirs['logs'] / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Print header
    logger.info("="*70)
    logger.info("SEQUENTIAL KNOWLEDGE DISTILLATION TRAINING")
    logger.info("="*70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {args.device}")

    # Check CUDA
    if args.device == 'cuda' and torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif args.device == 'cuda':
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'

    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # ============================================================
    # Load Dataset
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("LOADING DATASET")
    logger.info("="*70)

    dataset_config = config['dataset']
    train_loader, val_loader, dataset_info = create_dataloaders(
        data_dir=dataset_config['path'],
        image_size=dataset_config['image_size'],
        batch_size=dataset_config['batch_size'],
        train_ratio=dataset_config['train_split'],
        num_workers=dataset_config['num_workers']
    )

    # Extract class information
    class_names = dataset_info['class_names']
    num_classes = dataset_info['num_classes']

    # Save class mapping
    class_mapping = dataset_info['class_to_idx']
    with open(dirs['checkpoints'] / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")
    logger.info(f"Training samples: {dataset_info['train_samples']}")
    logger.info(f"Validation samples: {dataset_info['val_samples']}")

    # ============================================================
    # Create Student Model
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("CREATING ENHANCED STUDENT MODEL")
    logger.info("="*70)

    student_config = config['student']
    student_model = create_enhanced_student(
        num_classes=student_config['num_classes'],
        size=student_config.get('size', 'medium'),
        base_channels=student_config.get('base_channels', 48),
        expand_ratio=student_config.get('expand_ratio', 4),
        dropout_rate=student_config.get('dropout_rate', 0.3),
        num_consolidation_blocks=student_config.get('num_consolidation_blocks', 2),
        use_fpn=student_config.get('use_fpn', True)
    )

    logger.info(f"Student model size: {student_model.get_model_size_mb():.2f} MB")

    # ============================================================
    # Load Teacher Models
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("LOADING ALL 11 TEACHER MODELS")
    logger.info("="*70)

    teacher_config = config['teachers']
    teachers = MultiFormatTeacherEnsemble(
        model_configs=teacher_config['models'],
        models_dir=teacher_config['models_dir'],
        num_classes=teacher_config['teacher_num_classes'],
        device=args.device
    )

    logger.info(f"Loaded {teachers.get_teacher_count()} teachers")
    logger.info(f"Teacher order: {teachers.get_teacher_names()}")
    logger.info(f"Teacher formats: {teachers.get_teacher_formats()}")

    # ============================================================
    # Create Sequential Trainer
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING SEQUENTIAL TRAINER")
    logger.info("="*70)

    training_config = config['training']
    trainer = SequentialDistillationTrainer(
        config=training_config,
        student_model=student_model,
        teachers=teachers,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=args.device
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.best_accuracy = checkpoint.get('best_accuracy', 0)

    # ============================================================
    # Run Sequential Training
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("STARTING SEQUENTIAL TRAINING")
    logger.info("="*70)

    report = trainer.train()

    # ============================================================
    # Final Evaluation
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION")
    logger.info("="*70)

    # Load best model
    best_model_path = output_dir / 'student_model_final.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        student_model.load_state_dict(checkpoint['model_state_dict'])

    # Create evaluator with correct signature
    evaluator = ModelEvaluator(class_names, str(dirs['metrics']))

    # Evaluate model manually
    student_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(args.device)
            outputs = student_model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    import numpy as np
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(all_labels, all_preds, title="Final Student Model", save_name="final_confusion_matrix")

    # Calculate per-class accuracy
    per_class_acc = {}
    for i, cls_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[cls_name] = (all_preds[mask] == all_labels[mask]).mean()

    eval_results = {
        'accuracy': float(accuracy),
        'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()},
        'total_samples': len(all_labels)
    }

    logger.info(f"Final Accuracy: {accuracy:.2%}")
    logger.info(f"Per-class Accuracy:")
    for cls_name, acc in per_class_acc.items():
        logger.info(f"  {cls_name}: {acc:.2%}")

    # Save evaluation results
    eval_path = dirs['metrics'] / 'final_evaluation.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)

    # ============================================================
    # Export Models
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("EXPORTING MODELS")
    logger.info("="*70)

    export_config = config.get('export', {})
    exporter = ModelExporter(
        model=student_model,
        config=export_config,
        output_dir=dirs['exported_models'],
        class_names=class_names
    )

    export_results = exporter.export_all()

    for format_name, result in export_results.items():
        if result.get('success'):
            logger.info(f"  {format_name}: {result.get('path')}")
        else:
            logger.warning(f"  {format_name}: FAILED - {result.get('error')}")

    # ============================================================
    # Final Summary
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)

    logger.info(f"Best Accuracy: {report['summary']['best_accuracy']:.2%}")
    logger.info(f"Model Size: {student_model.get_model_size_mb():.2f} MB")
    logger.info(f"Teachers Used: {report['summary']['total_teachers']}")
    logger.info(f"\nPer-Teacher Results:")
    for teacher, results in report.get('per_teacher_results', {}).items():
        logger.info(f"  {teacher}: {results['best_accuracy']:.2%}")

    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"  - Checkpoints: {dirs['checkpoints']}")
    logger.info(f"  - Logs: {dirs['logs']}")
    logger.info(f"  - Models: {dirs['exported_models']}")
    logger.info(f"  - Metrics: {dirs['metrics']}")

    return report


if __name__ == '__main__':
    main()
