"""
Model Evaluator with Comprehensive Metrics and Visualization
=============================================================
Provides detailed evaluation including confusion matrices, 
classification reports, and training/validation curves.

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns  # type: ignore[import-unresolved]
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Install with: pip install seaborn")

try:
    from sklearn.metrics import (  # type: ignore[import-unresolved]
        confusion_matrix, 
        classification_report,
        precision_recall_fscore_support,
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations.
    
    Features:
    - Confusion matrix plotting
    - Classification report
    - Training/validation curves
    - Per-class accuracy analysis
    - Precision-Recall curves
    - Model comparison plots
    """
    
    def __init__(
        self,
        class_names: List[str],
        output_dir: str,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
            output_dir: Directory to save plots
            figsize: Default figure size
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = Path(output_dir)
        self.figsize = figsize
        
        # Create output directories
        self.plots_dir = self.output_dir / 'plots'
        self.metrics_dir = self.output_dir / 'metrics'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
        
        logger.info(f"Evaluator initialized with {self.num_classes} classes")
        logger.info(f"Plots directory: {self.plots_dir}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save_name: str = "confusion_matrix"
    ) -> str:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            normalize: Whether to normalize the matrix
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("matplotlib or sklearn not available, skipping confusion matrix")
            return ""
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_plot = cm_normalized
            fmt = '.2f'
            title_suffix = " (Normalized)"
        else:
            cm_plot = cm
            fmt = 'd'
            title_suffix = ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        if SEABORN_AVAILABLE:
            sns.heatmap(
                cm_plot, 
                annot=True, 
                fmt=fmt, 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
                annot_kws={'size': 10}
            )
        else:
            im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(cm_plot.shape[1]),
                yticks=np.arange(cm_plot.shape[0]),
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            
            # Add annotations
            thresh = cm_plot.max() / 2.
            for i in range(cm_plot.shape[0]):
                for j in range(cm_plot.shape[1]):
                    ax.text(j, i, format(cm_plot[i, j], fmt),
                           ha="center", va="center",
                           color="white" if cm_plot[i, j] > thresh else "black")
        
        plt.title(f"{title}{title_suffix}", fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        save_path = self.plots_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to: {save_path}")
        
        # Also save raw confusion matrix data
        np.save(self.metrics_dir / f"{save_name}_raw.npy", cm)
        
        return str(save_path)
    
    def plot_training_curves(
        self,
        history: List[Dict[str, Any]],
        save_name: str = "training_curves"
    ) -> str:
        """
        Plot training and validation curves.
        
        Args:
            history: List of epoch metrics
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping training curves")
            return ""
        
        if not history:
            logger.warning("No training history to plot")
            return ""
        
        epochs = range(1, len(history) + 1)
        
        # Extract metrics
        train_loss = [h.get('train_total_loss', h.get('train_loss', 0)) for h in history]
        val_loss = [h.get('val_total_loss', h.get('val_loss', 0)) for h in history]
        train_acc = [h.get('train_accuracy', 0) for h in history]
        val_acc = [h.get('val_accuracy', 0) for h in history]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curve
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate if available
        lr_values = [h.get('learning_rate', h.get('lr', None)) for h in history]
        if any(lr is not None for lr in lr_values):
            lr_values = [lr if lr is not None else 0 for lr in lr_values]
            axes[1, 0].plot(epochs, lr_values, 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # KD Loss components if available
        train_kd_loss = [h.get('train_kd_loss', None) for h in history]
        train_ce_loss = [h.get('train_ce_loss', None) for h in history]
        
        if any(kd is not None for kd in train_kd_loss):
            train_kd_loss = [kd if kd is not None else 0 for kd in train_kd_loss]
            train_ce_loss = [ce if ce is not None else 0 for ce in train_ce_loss]
            axes[1, 1].plot(epochs, train_kd_loss, 'c-', label='KD Loss', linewidth=2)
            axes[1, 1].plot(epochs, train_ce_loss, 'm-', label='CE Loss', linewidth=2)
            axes[1, 1].set_title('Loss Components', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Loss Components\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.suptitle('Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = self.plots_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to: {save_path}")
        
        return str(save_path)
    
    def plot_per_class_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "per_class_accuracy"
    ) -> str:
        """
        Plot per-class accuracy bar chart.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("matplotlib or sklearn not available, skipping per-class accuracy")
            return ""
        
        # Compute per-class accuracy
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Support (samples per class)
        support = cm.sum(axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.class_names))
        width = 0.6
        
        colors = plt.cm.RdYlGn(per_class_acc)
        bars = ax.bar(x, per_class_acc * 100, width, color=colors, edgecolor='black')
        
        # Add value labels on bars
        for bar, acc, sup in zip(bars, per_class_acc, support):
            height = bar.get_height()
            ax.annotate(f'{acc*100:.1f}%\n(n={sup})',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.axhline(y=np.mean(per_class_acc) * 100, color='r', linestyle='--', 
                   label=f'Mean: {np.mean(per_class_acc)*100:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        save_path = self.plots_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-class accuracy plot saved to: {save_path}")
        
        return str(save_path)
    
    def plot_precision_recall_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "precision_recall_f1"
    ) -> str:
        """
        Plot precision, recall, and F1 scores per class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("matplotlib or sklearn not available")
            return ""
        
        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes)
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#2196F3')
        bars2 = ax.bar(x, recall * 100, width, label='Recall', color='#4CAF50')
        bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='#FF9800')
        
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_title('Precision, Recall, and F1-Score per Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        save_path = self.plots_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision/Recall/F1 plot saved to: {save_path}")
        
        return str(save_path)
    
    def plot_class_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "class_distribution"
    ) -> str:
        """
        Plot true vs predicted class distribution.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available")
            return ""
        
        # Count distributions
        true_counts = np.bincount(y_true, minlength=self.num_classes)
        pred_counts = np.bincount(y_pred, minlength=self.num_classes)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, true_counts, width, label='True', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', color='#FF5722', alpha=0.8)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        save_path = self.plots_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class distribution plot saved to: {save_path}")
        
        return str(save_path)
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "classification_report"
    ) -> Dict[str, Any]:
        """
        Generate and save classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_name: Filename to save
            
        Returns:
            Classification report dictionary
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available")
            return {}
        
        # Generate report
        report_dict = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        report_str = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        
        # Save text report
        txt_path = self.metrics_dir / f"{save_name}.txt"
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(report_str)
            f.write("\n")
        
        # Save JSON report
        json_path = self.metrics_dir / f"{save_name}.json"
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Classification report saved to: {txt_path}")
        logger.info(f"Report JSON saved to: {json_path}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("CLASSIFICATION REPORT SUMMARY")
        logger.info("="*50)
        logger.info(f"Accuracy: {report_dict['accuracy']*100:.2f}%")
        logger.info(f"Macro Avg F1: {report_dict['macro avg']['f1-score']*100:.2f}%")
        logger.info(f"Weighted Avg F1: {report_dict['weighted avg']['f1-score']*100:.2f}%")
        
        return report_dict
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model and get predictions.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader for evaluation
            device: Device to use
            
        Returns:
            Tuple of (true labels, predicted labels, probabilities)
        """
        model.eval()
        model.to(device)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
    
    def run_full_evaluation(
        self,
        model: nn.Module,
        dataloader,
        training_history: List[Dict[str, Any]],
        device: str = "cuda",
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            model: Trained model
            dataloader: Validation/test dataloader
            training_history: Training history for curves
            device: Device to use
            prefix: Prefix for saved files
            
        Returns:
            Dictionary with all metrics and plot paths
        """
        logger.info("="*60)
        logger.info("RUNNING FULL MODEL EVALUATION")
        logger.info("="*60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'plots': {},
            'metrics': {}
        }
        
        # Get predictions
        logger.info("Evaluating model predictions...")
        y_true, y_pred, y_probs = self.evaluate_model(model, dataloader, device)
        
        # 1. Confusion Matrix
        logger.info("Generating confusion matrix...")
        cm_path = self.plot_confusion_matrix(
            y_true, y_pred,
            title="Student Model Confusion Matrix",
            normalize=True,
            save_name=f"{prefix}confusion_matrix_normalized" if prefix else "confusion_matrix_normalized"
        )
        results['plots']['confusion_matrix_normalized'] = cm_path
        
        cm_path_raw = self.plot_confusion_matrix(
            y_true, y_pred,
            title="Student Model Confusion Matrix (Raw Counts)",
            normalize=False,
            save_name=f"{prefix}confusion_matrix_raw" if prefix else "confusion_matrix_raw"
        )
        results['plots']['confusion_matrix_raw'] = cm_path_raw
        
        # 2. Training Curves
        logger.info("Generating training curves...")
        curves_path = self.plot_training_curves(
            training_history,
            save_name=f"{prefix}training_curves" if prefix else "training_curves"
        )
        results['plots']['training_curves'] = curves_path
        
        # 3. Per-class Accuracy
        logger.info("Generating per-class accuracy plot...")
        acc_path = self.plot_per_class_accuracy(
            y_true, y_pred,
            save_name=f"{prefix}per_class_accuracy" if prefix else "per_class_accuracy"
        )
        results['plots']['per_class_accuracy'] = acc_path
        
        # 4. Precision/Recall/F1
        logger.info("Generating precision/recall/F1 plot...")
        prf_path = self.plot_precision_recall_f1(
            y_true, y_pred,
            save_name=f"{prefix}precision_recall_f1" if prefix else "precision_recall_f1"
        )
        results['plots']['precision_recall_f1'] = prf_path
        
        # 5. Class Distribution
        logger.info("Generating class distribution plot...")
        dist_path = self.plot_class_distribution(
            y_true, y_pred,
            save_name=f"{prefix}class_distribution" if prefix else "class_distribution"
        )
        results['plots']['class_distribution'] = dist_path
        
        # 6. Classification Report
        logger.info("Generating classification report...")
        report = self.generate_classification_report(
            y_true, y_pred,
            save_name=f"{prefix}classification_report" if prefix else "classification_report"
        )
        results['metrics']['classification_report'] = report
        
        # 7. Overall metrics
        results['metrics']['accuracy'] = float(np.mean(y_true == y_pred))
        results['metrics']['num_samples'] = len(y_true)
        results['metrics']['num_correct'] = int(np.sum(y_true == y_pred))
        
        # Save results summary
        summary_path = self.metrics_dir / f"{prefix}evaluation_summary.json" if prefix else self.metrics_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {
                'timestamp': results['timestamp'],
                'plots': results['plots'],
                'metrics': {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in results['metrics'].items()
                }
            }
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info("="*60)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Accuracy: {results['metrics']['accuracy']*100:.2f}%")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)
        
        return results


def create_evaluator(
    class_names: List[str],
    output_dir: str
) -> ModelEvaluator:
    """
    Factory function to create evaluator.
    
    Args:
        class_names: List of class names
        output_dir: Output directory
        
    Returns:
        ModelEvaluator instance
    """
    return ModelEvaluator(class_names, output_dir)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("ModelEvaluator module loaded successfully")
    print(f"matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"seaborn available: {SEABORN_AVAILABLE}")
    print(f"sklearn available: {SKLEARN_AVAILABLE}")
