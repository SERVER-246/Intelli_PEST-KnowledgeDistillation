"""
Knowledge Distillation Training Script
=======================================
Main script to train a student model using knowledge distillation
from 11 teacher models.

Author: Knowledge Distillation Pipeline
Date: 2024-12-23

Usage:
    python train.py --config configs/config.yaml
    python train.py --data_dir "G:/AI work/IMAGE DATASET" --epochs 100
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from student_model import create_student_model, StudentCNN  # type: ignore[import-not-found]
from dataset import create_dataloaders, save_class_mapping  # type: ignore[import-not-found]
from trainer import TeacherEnsemble, KnowledgeDistillationTrainer  # type: ignore[import-not-found]
from exporter import ModelExporter  # type: ignore[import-not-found]
from evaluator import ModelEvaluator, create_evaluator  # type: ignore[import-not-found]

# Setup logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'main_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation Training for Pest Classification'
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    # Override options
    parser.add_argument('--data_dir', type=str, help='Override dataset directory')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    # Model options
    parser.add_argument(
        '--model_type', type=str, 
        choices=['standard', 'small', 'tiny'],
        default='standard',
        help='Student model type'
    )
    
    # Training options
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Export options
    parser.add_argument('--export_only', type=str, help='Only export from checkpoint')
    parser.add_argument('--no_export', action='store_true', help='Skip export after training')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    args = parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['dataset']['path'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.output_dir:
        config['paths']['root'] = args.output_dir
    
    output_dir = Path(config['paths']['root'])
    
    # Setup logging
    logger = setup_logging(str(output_dir))
    
    logger.info("="*70)
    logger.info("KNOWLEDGE DISTILLATION TRAINING")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {args.config}")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if device == "cuda":
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Running on CPU")
    
    # Export only mode
    if args.export_only:
        logger.info(f"Export only mode from checkpoint: {args.export_only}")
        export_from_checkpoint(args.export_only, config, output_dir, logger)
        return
    
    # Create dataloaders
    logger.info("\n" + "="*70)
    logger.info("LOADING DATASET")
    logger.info("="*70)
    
    train_loader, val_loader, dataset_info = create_dataloaders(
        data_dir=config['dataset']['path'],
        batch_size=config['dataset']['batch_size'],
        image_size=config['dataset']['image_size'],
        num_workers=config['dataset']['num_workers'],
        train_ratio=config['dataset']['train_split'],
        use_weighted_sampling=True,
        seed=args.seed
    )
    
    # Save class mapping
    class_mapping_path = output_dir / 'configs' / 'class_mapping.json'
    class_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    save_class_mapping(dataset_info['class_to_idx'], str(class_mapping_path))
    
    # Update config with actual dataset info
    config['dataset']['actual_classes'] = dataset_info['class_names']
    config['dataset']['actual_num_classes'] = dataset_info['num_classes']
    
    # Save updated config
    save_config(config, str(output_dir / 'configs' / 'training_config.yaml'))
    
    # Create student model
    logger.info("\n" + "="*70)
    logger.info("CREATING STUDENT MODEL")
    logger.info("="*70)
    
    student_model = create_student_model(
        model_type=args.model_type,
        num_classes=dataset_info['num_classes'],
        dropout_rate=config['student'].get('dropout_rate', 0.3)
    )
    
    model_size = student_model.get_model_size()
    logger.info(f"Student model size: {model_size:.2f} MB")
    
    if model_size > config['student'].get('target_size_mb', 20):
        logger.warning(f"Model size exceeds target of {config['student']['target_size_mb']} MB")
    
    # Create teacher ensemble
    logger.info("\n" + "="*70)
    logger.info("LOADING TEACHER MODELS")
    logger.info("="*70)
    
    teacher_ensemble = TeacherEnsemble(
        model_configs=config['teachers']['models'],
        models_dir=config['teachers']['models_dir'],
        teacher_num_classes=config['teachers']['teacher_num_classes'],
        device=device
    )
    
    # Create trainer
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING TRAINER")
    logger.info("="*70)
    
    trainer = KnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_ensemble=teacher_ensemble,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        dataset_info=dataset_info,
        output_dir=str(output_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_accuracy = checkpoint['best_val_accuracy']
    
    # Train
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    best_accuracy = trainer.train()
    
    # Export models
    if not args.no_export:
        logger.info("\n" + "="*70)
        logger.info("EXPORTING MODELS")
        logger.info("="*70)
        
        # Load best checkpoint
        best_checkpoint = torch.load(output_dir / 'checkpoints' / 'best_checkpoint.pt')
        student_model.load_state_dict(best_checkpoint['model_state_dict'])
        
        model_info = {
            'num_classes': dataset_info['num_classes'],
            'class_names': dataset_info['class_names'],
            'class_to_idx': dataset_info['class_to_idx'],
            'input_size': config['dataset']['image_size'],
            'model_type': args.model_type,
            'best_accuracy': best_accuracy,
            'training_date': datetime.now().isoformat()
        }
        
        exporter = ModelExporter(
            model=student_model,
            output_dir=str(output_dir / 'models' / 'exports'),
            model_info=model_info
        )
        
        export_results = exporter.export_all(
            export_pytorch=config['export']['pytorch']['enabled'],
            export_onnx=config['export']['onnx']['enabled'],
            export_tflite=config['export']['tflite']['enabled'],
            onnx_opset=config['export']['onnx']['opset_version'],
            tflite_quantization=config['export']['tflite']['quantization']
        )
        
        logger.info("\nExport Results:")
        for format_name, path in export_results.items():
            if path:
                size = Path(path).stat().st_size / (1024 * 1024)
                logger.info(f"  {format_name}: {path} ({size:.2f} MB)")
            else:
                logger.info(f"  {format_name}: FAILED")
    
    # Run comprehensive evaluation
    logger.info("\n" + "="*70)
    logger.info("RUNNING COMPREHENSIVE EVALUATION")
    logger.info("="*70)
    
    # Create evaluator
    evaluator = create_evaluator(
        class_names=dataset_info['class_names'],
        output_dir=str(output_dir)
    )
    
    # Load best model for evaluation
    best_checkpoint = torch.load(output_dir / 'checkpoints' / 'best_checkpoint.pt')
    student_model.load_state_dict(best_checkpoint['model_state_dict'])
    student_model.to(device)
    
    # Get training history from trainer
    training_history = trainer.training_history
    
    # Run full evaluation
    eval_results = evaluator.run_full_evaluation(
        model=student_model,
        dataloader=val_loader,
        training_history=training_history,
        device=device,
        prefix="final_"
    )
    
    # Save final evaluation results
    eval_summary_path = output_dir / 'evaluation_results.json'
    with open(eval_summary_path, 'w') as f:
        json.dump({
            'best_accuracy': float(best_accuracy),
            'final_evaluation': eval_results['metrics'],
            'plots_generated': list(eval_results['plots'].keys()),
            'model_info': {
                'type': args.model_type,
                'num_classes': dataset_info['num_classes'],
                'class_names': dataset_info['class_names'],
                'input_size': config['dataset']['image_size']
            }
        }, f, indent=2, default=str)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoints: {output_dir / 'checkpoints'}")
    logger.info(f"Metrics: {output_dir / 'metrics'}")
    logger.info(f"Plots: {output_dir / 'plots'}")
    logger.info(f"Exports: {output_dir / 'models' / 'exports'}")
    logger.info("="*70)


def export_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
):
    """Export model from a saved checkpoint."""
    
    checkpoint = torch.load(checkpoint_path)
    
    # Get model info from checkpoint
    dataset_info = checkpoint.get('dataset_info', {})
    num_classes = dataset_info.get('num_classes', config['student']['num_classes'])
    
    # Create model
    student_model = create_student_model(
        model_type='standard',
        num_classes=num_classes
    )
    
    # Load weights
    student_model.load_state_dict(checkpoint['model_state_dict'])
    
    model_info = {
        'num_classes': num_classes,
        'class_names': dataset_info.get('class_names', []),
        'class_to_idx': dataset_info.get('class_to_idx', {}),
        'input_size': config['dataset']['image_size'],
        'best_accuracy': checkpoint.get('best_val_accuracy', 0),
        'export_date': datetime.now().isoformat()
    }
    
    exporter = ModelExporter(
        model=student_model,
        output_dir=str(output_dir / 'models' / 'exports'),
        model_info=model_info
    )
    
    export_results = exporter.export_all(
        export_pytorch=True,
        export_onnx=True,
        export_tflite=True,
        onnx_opset=config['export']['onnx']['opset_version'],
        tflite_quantization=config['export']['tflite']['quantization']
    )
    
    logger.info("\nExport Results:")
    for format_name, path in export_results.items():
        if path:
            size = Path(path).stat().st_size / (1024 * 1024)
            logger.info(f"  {format_name}: {path} ({size:.2f} MB)")


if __name__ == '__main__':
    main()
