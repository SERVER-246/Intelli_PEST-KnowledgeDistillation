"""Analyze and compare model weights between good and bad models."""
import numpy as np
import torch

# Load good and bad models
good = torch.load('student_model_final.pth', map_location='cpu', weights_only=False)
bad = torch.load('student_model_v1.0.1.pt', map_location='cpu', weights_only=False)

good_state = good['model_state_dict']
bad_state = bad['model_state_dict']

print('=== Deep Layer Comparison ===')

# Compare classifier.4 (hidden layer before output)
good_w4 = good_state['classifier.4.weight'].numpy()
bad_w4 = bad_state['classifier.4.weight'].numpy()

print(f'classifier.4.weight shape: {good_w4.shape} vs {bad_w4.shape}')

# Check element-wise difference for hidden layers
max_diff = np.abs(good_w4 - bad_w4).max()
mean_diff = np.abs(good_w4 - bad_w4).mean()
print(f'classifier.4 weight max diff: {max_diff:.6f}')
print(f'classifier.4 weight mean diff: {mean_diff:.6f}')

# Compare classifier.1
good_w1 = good_state['classifier.1.weight'].numpy()
bad_w1 = bad_state['classifier.1.weight'].numpy()
max_diff1 = np.abs(good_w1 - bad_w1).max()
mean_diff1 = np.abs(good_w1 - bad_w1).mean()
print(f'classifier.1 weight max diff: {max_diff1:.6f}')
print(f'classifier.1 weight mean diff: {mean_diff1:.6f}')

print()
print('=== Intermediate features comparison (stage4) ===')
# Check a convolutional layer
good_conv = good_state['stage4.2.conv.3.weight'].numpy()
bad_conv = bad_state['stage4.2.conv.3.weight'].numpy()
max_diff_conv = np.abs(good_conv - bad_conv).max()
mean_diff_conv = np.abs(good_conv - bad_conv).mean()
print(f'stage4.2.conv.3 weight max diff: {max_diff_conv:.6f}')
print(f'stage4.2.conv.3 weight mean diff: {mean_diff_conv:.6f}')

print()
print('=== Class names in models ===')
if 'class_names' in good:
    print(f"Good model class names: {good['class_names']}")
elif 'model_config' in good:
    print(f"Good model_config: {good['model_config']}")
print(f"Bad model class names: {bad.get('class_names')}")

print()
print('=== Training info ===')
print(f"Good model training info:")
print(f"  best_accuracy: {good.get('best_accuracy')}")
print(f"  teachers_used: {good.get('teachers_used')}")

print(f"\nBad model training info:")
print(f"  training_type: {bad.get('training_type')}")
print(f"  training_metrics: {bad.get('training_metrics')}")
print(f"  total_comprehensive: {bad.get('total_comprehensive')}")
