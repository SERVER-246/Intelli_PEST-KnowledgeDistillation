"""
Test confidence of the properly expanded 12-class model.
Compare against the original 11-class model and the broken 12-class model.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "D:/KnowledgeDistillation/src")
from enhanced_student_model import EnhancedStudentModel

# Test images - using actual dataset location
TEST_IMAGES = [
    ("D:/IMAGE DATASET/Healthy/DSC03017.JPG", "Healthy"),
    ("D:/IMAGE DATASET/army worm/IMG_1891.JPG", "army worm"),
    ("D:/IMAGE DATASET/Top borer/DSC03013.JPG", "Top borer"),
    ("D:/IMAGE DATASET/termite/IMG_0916.JPG", "termite"),
    ("D:/IMAGE DATASET/Pink borer/DSC03022.JPG", "Pink borer"),
]

CLASS_NAMES_11 = [
    "Healthy", "Internode borer", "Pink borer", "Rat damage",
    "Stalk borer", "Top borer", "army worm", "mealy bug",
    "porcupine damage", "root borer", "termite"
]

CLASS_NAMES_12 = CLASS_NAMES_11 + ["junk"]

def load_model(model_path: str, num_classes: int, allow_mismatch: bool = False):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Get base_channels
    stem_weight = state_dict.get('stem.0.weight')
    base_channels = stem_weight.shape[0] if stem_weight is not None else 48

    # Detect actual num_classes from main classifier
    actual_classes = state_dict['classifier.6.weight'].shape[0]

    # Create model
    model = EnhancedStudentModel(num_classes=actual_classes, base_channels=base_channels)

    # Filter state dict to skip mismatched aux_classifiers
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'aux_classifiers' in key:
            # Check if shapes match
            model_param = dict(model.named_parameters()).get(key)
            if model_param is not None and model_param.shape != value.shape:
                print(f"    WARNING: Skipping mismatched {key}: {value.shape} vs {model_param.shape}")
                continue
        filtered_state_dict[key] = value

    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model, actual_classes

def predict(model, image_path: str, class_names: list):
    """Run prediction and return class name, confidence, and full probs."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        # Handle dict output (could be 'main' or 'logits')
        if isinstance(output, dict):
            output = output.get('main', output.get('logits', list(output.values())[0]))

        probs = F.softmax(output, dim=1)
        conf, pred_idx = probs.max(dim=1)

    return class_names[pred_idx.item()], conf.item(), probs[0].tolist()

def test_model(model_path: str, model_name: str, num_classes: int, class_names: list):
    """Test a single model."""
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"Expected Classes: {num_classes}")
    print('=' * 60)

    model, actual_classes = load_model(model_path, num_classes)
    actual_class_names = class_names[:actual_classes]

    print(f"Actual Classes: {actual_classes}")

    correct = 0
    total = 0

    for img_path, expected in TEST_IMAGES:
        if not Path(img_path).exists():
            print(f"  SKIP: {img_path} not found")
            continue

        pred_class, confidence, probs = predict(model, img_path, actual_class_names)
        is_correct = pred_class == expected
        correct += int(is_correct)
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"  {status} {expected:20s} → {pred_class:20s} ({confidence*100:5.1f}%)")

        # If 12-class model, show junk probability
        if actual_classes == 12:
            junk_prob = probs[11]
            print(f"      (junk prob: {junk_prob*100:5.2f}%)")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    return accuracy

def main():
    print("=" * 60)
    print("CONFIDENCE COMPARISON TEST")
    print("=" * 60)

    models_to_test = [
        ("D:/KnowledgeDistillation/student_model_final.pth", "Original 11-class (GOOD)", 11, CLASS_NAMES_11),
        ("D:/KnowledgeDistillation/student_model_12class_proper.pt", "Properly Expanded 12-class", 12, CLASS_NAMES_12),
        ("D:/KnowledgeDistillation/student_model_v1.0.1.pt", "Broken 12-class (v1.0.1)", 12, CLASS_NAMES_12),
    ]

    results = {}
    for model_path, model_name, num_classes, class_names in models_to_test:
        if not Path(model_path).exists():
            print(f"\nSKIP: {model_path} not found")
            continue

        acc = test_model(model_path, model_name, num_classes, class_names)
        results[model_name] = acc

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name}: {acc:.1f}%")

if __name__ == "__main__":
    main()
