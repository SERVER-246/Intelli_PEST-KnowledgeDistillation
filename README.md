# Knowledge Distillation for Pest Classification

Multi-teacher knowledge distillation pipeline to create a lightweight, efficient model for pest classification using 11 pre-trained teacher models.

## 📁 Project Structure

```
KnowledgeDistillation/
├── configs/
│   ├── config.yaml          # Main configuration file
│   └── class_mapping.json   # Class name to index mapping
├── src/
│   ├── __init__.py
│   ├── student_model.py     # Lightweight CNN architecture
│   ├── dataset.py           # Data loading and augmentation
│   ├── trainer.py           # Knowledge distillation trainer
│   └── exporter.py          # Model export utilities
├── models/
│   ├── student/             # Student model checkpoints
│   └── exports/
│       ├── pytorch/         # PyTorch exports (.pt)
│       ├── onnx/            # ONNX exports (.onnx)
│       └── tflite/          # TFLite exports (.tflite)
├── checkpoints/             # Training checkpoints
├── logs/                    # Training logs
├── metrics/                 # Training metrics (JSON)
├── train.py                 # Main training script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Training

Edit `configs/config.yaml` to set:
- Dataset path
- Training hyperparameters
- Teacher model locations
- Export settings

### 3. Run Training

```bash
# Use default config
python train.py

# Override settings
python train.py --data_dir "path/to/dataset" --epochs 100 --batch_size 32

# Use smaller model for mobile
python train.py --model_type small

# Resume training
python train.py --resume checkpoints/latest_checkpoint.pt
```

### 4. Export Only

```bash
python train.py --export_only checkpoints/best_checkpoint.pt
```

## 📊 Dataset Format

The dataset should be organized in ImageFolder format:
```
IMAGE DATASET/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   └── ...
└── ...
```

## 🎓 Teacher Models

Uses 11 pre-trained ONNX models:
- **Base models**: MobileNetV2, ResNet50, InceptionV3, EfficientNet-B0, DarkNet53, AlexNet, YOLO11n-cls
- **Ensemble models**: Attention, Concat, Cross, Super Ensemble

## 📈 Training Features

- **Knowledge Distillation**: Soft labels from teacher ensemble
- **Temperature Scaling**: Soften probability distributions
- **Weighted Ensemble**: Higher weight for ensemble teachers
- **Mixed Precision**: FP16 training for faster computation
- **Data Augmentation**: Rotation, flip, color jitter, etc.
- **Weighted Sampling**: Handle class imbalance
- **Early Stopping**: Prevent overfitting
- **Comprehensive Logging**: Track all metrics

## 🔧 Student Model Architecture

Custom lightweight CNN with:
- Depthwise separable convolutions
- Squeeze-and-Excitation blocks
- Residual connections
- Global average pooling

**Model Variants:**
| Variant  | Size  | Parameters |
|----------|-------|------------|
| Standard | ~15MB | ~4M        |
| Small    | ~8MB  | ~2M        |
| Tiny     | ~3MB  | ~800K      |

## 📦 Export Formats

- **PyTorch** (.pt): Full model with weights
- **ONNX** (.onnx): Opset 13 for cross-platform
- **TFLite** (.tflite): TF 2.14 compatible for Android

## 📋 Metrics Tracked

- Training/Validation Loss
- Accuracy (overall and per-class)
- Precision, Recall, F1 Score
- Confusion Matrix
- Learning Rate

## 📄 Output Files

After training:
- `checkpoints/best_checkpoint.pt` - Best model
- `metrics/training_history.json` - Full training history
- `models/exports/` - Exported models
- `logs/` - Training logs
- `training_summary.json` - Final summary

## 🔬 Configuration Options

See `configs/config.yaml` for all options:

```yaml
distillation:
  temperature: 4.0    # Softening temperature
  alpha: 0.7          # Weight for soft labels
  beta: 0.3           # Weight for hard labels

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adamw
  scheduler: cosine
```

## 📝 License

MIT License

## 👥 Authors

Knowledge Distillation Pipeline - December 2024
