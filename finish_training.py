"""
Finish Training - Run final evaluation and export
"""
import torch
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')


def main():
    from src.enhanced_student_model import create_enhanced_student
    from src.dataset import create_dataloaders
    from src.evaluator import ModelEvaluator
    import yaml

    print('Loading model and data...')

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create data loader with 0 workers to avoid multiprocessing issues
    dataset_config = config['dataset']
    train_loader, val_loader, dataset_info = create_dataloaders(
        data_dir=dataset_config['path'],
        image_size=dataset_config['image_size'],
        batch_size=dataset_config['batch_size'],
        train_ratio=dataset_config['train_split'],
        num_workers=0  # Use 0 workers for Windows compatibility
    )
    class_names = dataset_info['class_names']

    # Create and load model
    student_config = config['student']
    student_model = create_enhanced_student(
        num_classes=student_config['num_classes'],
        size=student_config.get('size', 'medium')
    )

    # Load trained weights
    checkpoint = torch.load('student_model_final.pth')
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.eval()
    student_model.cuda()

    best_acc = checkpoint.get('best_accuracy', 'N/A')
    print(f'Best accuracy from training: {best_acc}')

    # Evaluate
    print('\nEvaluating model...')
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            outputs = student_model(images)
            # Handle dict output (model might return dict with 'logits' key)
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print(f'Overall Accuracy: {accuracy:.2%}')
    print('\nPer-class Accuracy:')
    per_class_acc = {}
    for i, cls_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_labels[mask]).mean()
            per_class_acc[cls_name] = float(cls_acc)
            print(f'  {cls_name}: {cls_acc:.2%}')

    # Create confusion matrix plot
    evaluator = ModelEvaluator(class_names, 'metrics')
    evaluator.plot_confusion_matrix(all_labels, all_preds, title='Final Student Model', save_name='final_confusion_matrix')
    print('\nConfusion matrix saved to metrics/plots/')

    # Save evaluation results
    eval_results = {
        'accuracy': float(accuracy),
        'per_class_accuracy': per_class_acc,
        'total_samples': len(all_labels),
        'class_names': class_names
    }
    Path('metrics').mkdir(exist_ok=True)
    with open('metrics/final_evaluation.json', 'w') as f:
        json.dump(eval_results, f, indent=2)

    # Export models
    print('\n' + '='*60)
    print('EXPORTING MODELS')
    print('='*60)
    export_dir = Path('exported_models')
    export_dir.mkdir(exist_ok=True)

    # Move to CPU for export
    student_model.cpu()

    # PyTorch export
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'class_names': class_names,
        'accuracy': float(accuracy)
    }, export_dir / 'student_model.pt')
    print(f'PyTorch model saved: {export_dir}/student_model.pt')

    # ONNX export
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        student_model,
        dummy_input,
        str(export_dir / 'student_model.onnx'),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=13
    )
    print(f'ONNX model saved: {export_dir}/student_model.onnx')

    # TFLite export
    print('\nConverting to TFLite...')
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(str(export_dir / 'student_model.onnx'))
        tf_rep = prepare(onnx_model)
        
        # Export to SavedModel
        tf_rep.export_graph(str(export_dir / 'tf_saved_model'))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir / 'tf_saved_model'))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(export_dir / 'student_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print(f'TFLite model saved: {export_dir}/student_model.tflite')
    except Exception as e:
        print(f'TFLite conversion failed: {e}')
        print('You can convert manually later.')

    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print('='*60)
    print(f'Final Validation Accuracy: {accuracy:.2%}')
    print('\nExported models:')
    for file in export_dir.iterdir():
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f'  {file.name}: {size_mb:.2f} MB')


if __name__ == '__main__':
    main()
