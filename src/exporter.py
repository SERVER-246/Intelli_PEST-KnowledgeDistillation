"""
Model Exporter for Knowledge Distillation
==========================================
Export trained student model to PyTorch, ONNX, and TFLite formats.

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def export_pytorch(
    model: nn.Module,
    output_path: str,
    model_info: Dict[str, Any]
) -> str:
    """
    Export model to PyTorch format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save the model
        model_info: Model metadata
        
    Returns:
        Path to saved model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full model (state dict + architecture info)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_info': model_info,
        'num_classes': model_info.get('num_classes', 12),
        'input_size': model_info.get('input_size', 256)
    }
    
    torch.save(save_dict, output_path)
    
    # Also save just the state dict
    state_dict_path = output_path.parent / f"{output_path.stem}_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"PyTorch model saved to: {output_path}")
    logger.info(f"  Model size: {size_mb:.2f} MB")
    
    return str(output_path)


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_size: int = 256,
    opset_version: int = 13,
    dynamic_axes: bool = False
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save the model
        input_size: Input image size
        opset_version: ONNX opset version (13 recommended for compatibility)
        dynamic_axes: Whether to use dynamic batch size
        
    Returns:
        Path to saved model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    if torch.cuda.is_available():
        model = model.cpu()
        dummy_input = dummy_input.cpu()
    
    # Dynamic axes configuration
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_config = None
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_config
    )
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified successfully")
    except ImportError:
        logger.warning("onnx package not installed, skipping verification")
    except Exception as e:
        logger.warning(f"ONNX verification warning: {e}")
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"ONNX model saved to: {output_path}")
    logger.info(f"  Model size: {size_mb:.2f} MB")
    logger.info(f"  Opset version: {opset_version}")
    
    return str(output_path)


def export_tflite(
    onnx_path: str,
    output_path: str,
    quantization: str = "dynamic",
    input_size: int = 256
) -> str:
    """
    Export model to TFLite format (TF 2.14 compatible).
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TFLite model
        quantization: Quantization type ('none', 'dynamic', 'float16', 'int8')
        input_size: Input image size
        
    Returns:
        Path to saved model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import tensorflow as tf  # type: ignore[import-unresolved]
        import onnx
        from onnx_tf.backend import prepare  # type: ignore[import-unresolved]
        
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Create a temporary SavedModel directory
        temp_saved_model_dir = output_path.parent / "temp_saved_model"
        tf_rep.export_graph(str(temp_saved_model_dir))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model_dir))
        
        # Set optimizations based on quantization type
        if quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Note: Full int8 quantization requires representative dataset
            logger.warning("Full int8 quantization requires representative dataset")
        
        # Set target ops for better compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Clean up temp directory
        shutil.rmtree(temp_saved_model_dir, ignore_errors=True)
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"TFLite model saved to: {output_path}")
        logger.info(f"  Model size: {size_mb:.2f} MB")
        logger.info(f"  Quantization: {quantization}")
        
        return str(output_path)
        
    except ImportError as e:
        logger.error(f"Missing dependencies for TFLite conversion: {e}")
        logger.error("Install with: pip install tensorflow onnx onnx-tf")
        raise
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        raise


def export_tflite_direct(
    model: nn.Module,
    output_path: str,
    input_size: int = 256,
    quantization: str = "dynamic"
) -> str:
    """
    Alternative TFLite export using torch -> onnx -> tf -> tflite pipeline.
    Uses more compatible operators.
    
    Args:
        model: PyTorch model
        output_path: Path to save TFLite model
        input_size: Input image size
        quantization: Quantization type
        
    Returns:
        Path to saved model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import tensorflow as tf  # type: ignore[import-unresolved]
        import onnx
        from onnx_tf.backend import prepare  # type: ignore[import-unresolved]
        import tempfile
        
        model.eval()
        model = model.cpu()
        
        # Create temp ONNX file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_onnx_path = f.name
        
        # Export to ONNX with compatibility settings
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        torch.onnx.export(
            model,
            dummy_input,
            temp_onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # Use simpler ops for better TFLite compatibility
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
        )
        
        # Load and simplify ONNX
        onnx_model = onnx.load(temp_onnx_path)
        
        try:
            from onnxsim import simplify  # type: ignore
            onnx_model, check = simplify(onnx_model)
            if check:
                logger.info("ONNX model simplified successfully")
        except ImportError:
            logger.warning("onnxsim not available, skipping simplification")
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        temp_saved_model = output_path.parent / "temp_tf_model"
        tf_rep.export_graph(str(temp_saved_model))
        
        # Convert to TFLite with compatibility settings
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model))
        
        # Optimizations
        if quantization != "none":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
        
        # Use TFLite built-in ops only for maximum compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        
        # Experimental settings for better compatibility
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        
        try:
            tflite_model = converter.convert()
        except Exception as e:
            logger.warning(f"Strict conversion failed, trying with SELECT_TF_OPS: {e}")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Cleanup
        os.unlink(temp_onnx_path)
        shutil.rmtree(temp_saved_model, ignore_errors=True)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"TFLite model saved to: {output_path}")
        logger.info(f"  Model size: {size_mb:.2f} MB")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Direct TFLite conversion failed: {e}")
        raise


def verify_tflite_model(model_path: str, input_size: int = 256) -> Dict[str, Any]:
    """
    Verify TFLite model and return info.
    
    Args:
        model_path: Path to TFLite model
        input_size: Expected input size
        
    Returns:
        Model information dictionary
    """
    try:
        import tensorflow as tf  # type: ignore[import-unresolved]
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test inference
        input_shape = input_details[0]['shape']
        input_data = np.random.rand(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        info = {
            'valid': True,
            'input_shape': input_shape.tolist(),
            'output_shape': output_data.shape,
            'input_dtype': str(input_details[0]['dtype']),
            'output_dtype': str(output_details[0]['dtype']),
            'num_classes': output_data.shape[-1]
        }
        
        logger.info(f"TFLite model verification:")
        logger.info(f"  Input shape: {info['input_shape']}")
        logger.info(f"  Output shape: {info['output_shape']}")
        logger.info(f"  Num classes: {info['num_classes']}")
        
        return info
        
    except Exception as e:
        logger.error(f"TFLite verification failed: {e}")
        return {'valid': False, 'error': str(e)}


class ModelExporter:
    """
    Complete model export pipeline.
    """
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        model_info: Dict[str, Any]
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.model_info = model_info
        self.export_results = {}
    
    def export_all(
        self,
        export_pytorch: bool = True,
        export_onnx: bool = True,
        export_tflite: bool = True,
        onnx_opset: int = 13,
        tflite_quantization: str = "dynamic"
    ) -> Dict[str, str]:
        """
        Export model to all formats.
        
        Returns:
            Dictionary mapping format to file path
        """
        input_size = self.model_info.get('input_size', 256)
        
        results = {}
        
        # PyTorch export
        if export_pytorch:
            try:
                pytorch_path = self.output_dir / "pytorch" / "student_model.pt"
                results['pytorch'] = export_pytorch_model(
                    self.model, str(pytorch_path), self.model_info
                )
                logger.info("✓ PyTorch export successful")
            except Exception as e:
                logger.error(f"✗ PyTorch export failed: {e}")
                results['pytorch'] = None
        
        # ONNX export
        if export_onnx:
            try:
                onnx_path = self.output_dir / "onnx" / "student_model.onnx"
                results['onnx'] = export_onnx_model(
                    self.model, str(onnx_path), input_size, onnx_opset
                )
                logger.info("✓ ONNX export successful")
            except Exception as e:
                logger.error(f"✗ ONNX export failed: {e}")
                results['onnx'] = None
        
        # TFLite export
        if export_tflite and results.get('onnx'):
            try:
                tflite_path = self.output_dir / "tflite" / "student_model.tflite"
                results['tflite'] = export_tflite(
                    results['onnx'], str(tflite_path), tflite_quantization, input_size
                )
                
                # Verify
                verify_info = verify_tflite_model(str(tflite_path), input_size)
                if verify_info.get('valid'):
                    logger.info("✓ TFLite export and verification successful")
                else:
                    logger.warning("⚠ TFLite export successful but verification failed")
                    
            except Exception as e:
                logger.error(f"✗ TFLite export failed: {e}")
                # Try direct conversion
                try:
                    logger.info("Attempting direct TFLite conversion...")
                    tflite_path = self.output_dir / "tflite" / "student_model.tflite"
                    results['tflite'] = export_tflite_direct(
                        self.model, str(tflite_path), input_size, tflite_quantization
                    )
                    logger.info("✓ Direct TFLite export successful")
                except Exception as e2:
                    logger.error(f"✗ Direct TFLite export also failed: {e2}")
                    results['tflite'] = None
        
        # Save export summary
        self._save_export_summary(results)
        
        return results
    
    def _save_export_summary(self, results: Dict[str, str]):
        """Save export summary to JSON."""
        summary = {
            'export_time': str(Path(__file__).stat().st_mtime),
            'model_info': self.model_info,
            'exports': {}
        }
        
        for format_name, path in results.items():
            if path and Path(path).exists():
                summary['exports'][format_name] = {
                    'path': path,
                    'size_mb': Path(path).stat().st_size / (1024 * 1024)
                }
        
        with open(self.output_dir / 'export_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# Alias functions for cleaner API
def export_pytorch_model(model, output_path, model_info):
    return export_pytorch(model, output_path, model_info)

def export_onnx_model(model, output_path, input_size=256, opset_version=13):
    return export_onnx(model, output_path, input_size, opset_version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Model exporter module loaded successfully")
