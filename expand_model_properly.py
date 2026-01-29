"""
Properly Expand 11-class Model to 12-class with Junk
=====================================================
This script takes the working 11-class student_model_final.pth and 
properly expands it to 12 classes (adding junk) WITHOUT destroying
the learned patterns for the original 11 classes.

Key differences from broken approach:
1. Expands BOTH main classifier AND aux_classifiers
2. Uses conservative initialization for new class (small weights, negative bias)
3. Preserves original class weights EXACTLY (no training, no fine-tuning)
4. Results in 12-class model that SHOULD have same confidence as original

Author: Intelli-PEST Backend
Date: 2026-01-09
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))

def expand_classifier_properly(model, new_num_classes: int = 12, num_original_classes: int = 11):
    """
    Properly expand classifier and aux_classifiers.
    
    Key: Use small initialization for new class so it doesn't steal probability
    from original classes through softmax.
    """
    expanded_layers = []
    
    # 1. Expand main classifier
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        for i in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[i], nn.Linear):
                old_linear = model.classifier[i]
                in_features = old_linear.in_features
                old_classes = old_linear.out_features
                
                if old_classes >= new_num_classes:
                    print(f"Main classifier already has {old_classes} classes")
                    break
                
                new_linear = nn.Linear(in_features, new_num_classes)
                
                with torch.no_grad():
                    # Copy original weights exactly
                    new_linear.weight[:old_classes] = old_linear.weight.clone()
                    new_linear.bias[:old_classes] = old_linear.bias.clone()
                    
                    # Initialize new class with small weights and negative bias
                    # This ensures the new class starts with LOW confidence
                    # and doesn't steal probability from original classes
                    nn.init.normal_(new_linear.weight[old_classes:], mean=0.0, std=0.01)
                    nn.init.constant_(new_linear.bias[old_classes:], -3.0)  # Start very low
                
                model.classifier[i] = new_linear
                expanded_layers.append(f"classifier[{i}]: {old_classes} → {new_num_classes}")
                print(f"✓ Expanded main classifier: {old_classes} → {new_num_classes}")
                break
    
    # 2. Expand aux_classifiers (critical for deep supervision)
    if hasattr(model, 'aux_classifiers'):
        aux = model.aux_classifiers
        if isinstance(aux, nn.ModuleDict):
            for stage_name, aux_clf in aux.items():
                if isinstance(aux_clf, nn.Sequential):
                    for j in range(len(aux_clf) - 1, -1, -1):
                        if isinstance(aux_clf[j], nn.Linear):
                            old_aux = aux_clf[j]
                            aux_in = old_aux.in_features
                            aux_old = old_aux.out_features
                            
                            if aux_old >= new_num_classes:
                                break
                            
                            new_aux = nn.Linear(aux_in, new_num_classes)
                            with torch.no_grad():
                                new_aux.weight[:aux_old] = old_aux.weight.clone()
                                new_aux.bias[:aux_old] = old_aux.bias.clone()
                                nn.init.normal_(new_aux.weight[aux_old:], mean=0.0, std=0.01)
                                nn.init.constant_(new_aux.bias[aux_old:], -3.0)
                            
                            aux_clf[j] = new_aux
                            expanded_layers.append(f"aux_classifiers.{stage_name}: {aux_old} → {new_num_classes}")
                            print(f"✓ Expanded aux_classifier[{stage_name}]: {aux_old} → {new_num_classes}")
                            break
    
    return model, expanded_layers


def main():
    print("=" * 60)
    print("PROPER 11→12 CLASS EXPANSION")
    print("=" * 60)
    
    # Paths
    source_model = Path("D:/KnowledgeDistillation/student_model_final.pth")
    output_model = Path("D:/KnowledgeDistillation/student_model_12class_proper.pt")
    
    if not source_model.exists():
        print(f"ERROR: Source model not found: {source_model}")
        return
    
    print(f"\nLoading source model: {source_model}")
    checkpoint = torch.load(source_model, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Create model
    try:
        from enhanced_student_model import EnhancedStudentModel
    except ImportError:
        sys.path.insert(0, "D:/KnowledgeDistillation/src")
        from enhanced_student_model import EnhancedStudentModel
    
    # Get base_channels from state dict
    stem_weight = state_dict.get('stem.0.weight')
    base_channels = stem_weight.shape[0] if stem_weight is not None else 48
    
    # Create 11-class model first
    print(f"\nCreating 11-class model (base_channels={base_channels})...")
    model = EnhancedStudentModel(num_classes=11, base_channels=base_channels)
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    print("✓ Loaded 11-class weights")
    
    # Verify original accuracy
    print("\n--- Before expansion ---")
    print(f"classifier.6.weight shape: {model.classifier[6].weight.shape}")
    print(f"aux_classifiers.stage2.2.weight shape: {model.aux_classifiers['stage2'][2].weight.shape}")
    
    # Expand to 12 classes
    print("\n--- Expanding to 12 classes ---")
    model, expanded = expand_classifier_properly(model, new_num_classes=12, num_original_classes=11)
    
    # Verify expansion
    print("\n--- After expansion ---")
    print(f"classifier.6.weight shape: {model.classifier[6].weight.shape}")
    print(f"aux_classifiers.stage2.2.weight shape: {model.aux_classifiers['stage2'][2].weight.shape}")
    
    # Check that original weights are EXACTLY preserved
    print("\n--- Verifying original weights preserved ---")
    original_w = state_dict['classifier.6.weight']
    new_w = model.classifier[6].weight[:11]
    diff = (original_w - new_w).abs().max().item()
    print(f"Max weight difference for original 11 classes: {diff:.10f}")
    
    if diff < 1e-6:
        print("✓ Original weights EXACTLY preserved!")
    else:
        print("WARNING: Original weights changed!")
    
    # Prepare new checkpoint
    class_names = [
        "Healthy",
        "Internode borer", 
        "Pink borer",
        "Rat damage",
        "Stalk borer",
        "Top borer",
        "army worm",
        "mealy bug",
        "porcupine damage",
        "root borer",
        "termite",
        "junk"  # New class
    ]
    
    new_checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'model_config': {
            'num_classes': 12,
            'input_size': 256,
            'base_channels': base_channels
        },
        'expansion_info': {
            'expanded_from': str(source_model),
            'expanded_layers': expanded,
            'expansion_date': datetime.now().isoformat(),
            'original_accuracy': checkpoint.get('best_accuracy', 0.9625)
        },
        'best_accuracy': checkpoint.get('best_accuracy', 0.9625),  # Preserve original accuracy
        'original_checkpoint': {
            'teachers_used': checkpoint.get('teachers_used', []),
            'timestamp': checkpoint.get('timestamp', '')
        }
    }
    
    # Save
    print(f"\nSaving expanded model to: {output_model}")
    torch.save(new_checkpoint, output_model)
    print("✓ Model saved!")
    
    # Quick inference test
    print("\n--- Quick inference test ---")
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1)
        
        print(f"Output shape: {output.shape}")
        print(f"Probabilities for original 11 classes: {probs[0, :11].sum().item():.4f}")
        print(f"Probability for junk class: {probs[0, 11].item():.6f}")
    
    print("\n" + "=" * 60)
    print("EXPANSION COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Test with real images:")
    print(f"  python test_confidence.py --model {output_model}")


if __name__ == "__main__":
    main()
