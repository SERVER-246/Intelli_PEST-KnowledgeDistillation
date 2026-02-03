"""
Knowledge Distillation Package
==============================
Multi-teacher knowledge distillation for pest classification.
"""

from .dataset import PestDataset, create_dataloaders
from .evaluator import ModelEvaluator, create_evaluator
from .exporter import ModelExporter
from .student_model import StudentCNN, create_student_model
from .trainer import KnowledgeDistillationTrainer, TeacherEnsemble

__version__ = "1.0.0"
__author__ = "Knowledge Distillation Pipeline"

__all__ = [
    'StudentCNN',
    'create_student_model',
    'PestDataset',
    'create_dataloaders',
    'TeacherEnsemble',
    'KnowledgeDistillationTrainer',
    'ModelExporter',
    'ModelEvaluator',
    'create_evaluator'
]
