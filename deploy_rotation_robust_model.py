#!/usr/bin/env python3
"""
Deployment and Testing Script for Rotation-Robust Student Model

This script:
1. Checks if training has completed
2. Backs up the current model
3. Deploys the new rotation-robust model
4. Runs comprehensive tests with rotated images
5. Generates a deployment report

Usage:
    python deploy_rotation_robust_model.py [--force] [--skip-tests]
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    # Model paths
    "new_model_path": r"D:\KnowledgeDistillation\exported_models\rotation_robust_student.pt",
    "current_model_path": r"D:\KnowledgeDistillation\exported_models\student_model.pt",
    "backup_dir": r"D:\KnowledgeDistillation\exported_models\backups",

    # Test images directory
    "test_images_dir": r"D:\Test-images",

    # Server config
    "server_url": "http://localhost:8000",
    "tunnel_url": "https://cnbrr7xn-8000.inc1.devtunnels.ms",
    "api_key": "ip_test_key_intelli_pest_2025",

    # Class names
    "class_names": [
        "Healthy", "Internode borer", "Pink borer", "Rat damage",
        "Stalk borer", "Top borer", "army worm", "mealy bug",
        "porcupine damage", "root borer", "termite"
    ],

    # Test rotations
    "test_rotations": [0, 90, 180, 270],
}


# ============================================================
# Student Model Architecture (must match training)
# ============================================================

class StudentCNN(nn.Module):
    """Compact CNN for mobile deployment - must match training architecture"""

    def __init__(self, num_classes: int = 11):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# Utility Functions
# ============================================================

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_status(message: str, status: str = "INFO"):
    """Print a status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "WAIT": "⏳"}
    symbol = symbols.get(status, "•")
    print(f"[{timestamp}] {symbol} {message}")


def check_training_complete() -> dict:
    """Check if training has completed and return info about the model"""
    new_model_path = CONFIG["new_model_path"]

    result = {
        "exists": False,
        "path": new_model_path,
        "size_mb": 0,
        "modified_time": None,
        "ready": False,
    }

    if os.path.exists(new_model_path):
        result["exists"] = True
        stat = os.stat(new_model_path)
        result["size_mb"] = stat.st_size / (1024 * 1024)
        result["modified_time"] = datetime.fromtimestamp(stat.st_mtime)

        # Check if file is still being written (size > 1MB and recent)
        if result["size_mb"] > 1:
            result["ready"] = True

    return result


def backup_current_model() -> str:
    """Backup the current model with timestamp"""
    current_path = CONFIG["current_model_path"]
    backup_dir = CONFIG["backup_dir"]

    if not os.path.exists(current_path):
        print_status(f"No current model to backup at {current_path}", "WARN")
        return None

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"student_model_backup_{timestamp}.pt"
    backup_path = os.path.join(backup_dir, backup_name)

    # Copy the file
    shutil.copy2(current_path, backup_path)
    print_status(f"Backed up current model to: {backup_path}", "OK")

    return backup_path


def deploy_new_model() -> bool:
    """Deploy the new rotation-robust model"""
    new_path = CONFIG["new_model_path"]
    current_path = CONFIG["current_model_path"]

    if not os.path.exists(new_path):
        print_status(f"New model not found at {new_path}", "ERROR")
        return False

    # Copy new model to current location
    shutil.copy2(new_path, current_path)
    print_status(f"Deployed new model to: {current_path}", "OK")

    return True


def load_model(model_path: str, device: str = "cuda") -> nn.Module:
    """Load a PyTorch model"""
    model = StudentCNN(num_classes=len(CONFIG["class_names"]))

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def get_transform():
    """Get the image transform for inference"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def predict_image(model: nn.Module, image_path: str, rotation: int = 0, device: str = "cuda") -> dict:
    """Run prediction on a single image with optional rotation"""
    transform = get_transform()

    # Load and rotate image
    image = Image.open(image_path).convert("RGB")
    if rotation != 0:
        image = image.rotate(-rotation, expand=True)  # PIL rotates counter-clockwise

    # Transform and predict
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "class_idx": predicted.item(),
        "class_name": CONFIG["class_names"][predicted.item()],
        "confidence": confidence.item() * 100,
        "rotation": rotation,
    }


def test_model_locally(model_path: str) -> dict:
    """Test model locally with test images at different rotations"""
    print_header("Local Model Testing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_status(f"Using device: {device}", "INFO")

    # Load model
    model = load_model(model_path, device)
    print_status("Model loaded successfully", "OK")

    test_dir = CONFIG["test_images_dir"]
    results = {
        "total_tests": 0,
        "rotation_consistent": 0,
        "by_class": {},
        "detailed_results": [],
    }

    if not os.path.exists(test_dir):
        print_status(f"Test directory not found: {test_dir}", "ERROR")
        return results

    # Test each class
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print_status(f"\nTesting class: {class_name}", "INFO")
        results["by_class"][class_name] = {
            "total": 0,
            "rotation_consistent": 0,
            "correct_at_rotations": {r: 0 for r in CONFIG["test_rotations"]},
        }

        # Get test images
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]  # Test up to 3 images per class

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            predictions = []

            for rotation in CONFIG["test_rotations"]:
                pred = predict_image(model, img_path, rotation, device)
                predictions.append(pred)

                # Check if prediction matches expected class
                expected_class = class_name.lower().strip()
                predicted_class = pred["class_name"].lower().strip()

                # Fuzzy match for class names
                is_correct = (expected_class in predicted_class or
                             predicted_class in expected_class or
                             expected_class.replace(" ", "") == predicted_class.replace(" ", ""))

                if is_correct:
                    results["by_class"][class_name]["correct_at_rotations"][rotation] += 1

            # Check rotation consistency (same prediction for all rotations)
            unique_predictions = set(p["class_name"] for p in predictions)
            is_consistent = len(unique_predictions) == 1

            results["total_tests"] += 1
            results["by_class"][class_name]["total"] += 1

            if is_consistent:
                results["rotation_consistent"] += 1
                results["by_class"][class_name]["rotation_consistent"] += 1

            results["detailed_results"].append({
                "image": img_name,
                "class": class_name,
                "predictions": predictions,
                "rotation_consistent": is_consistent,
            })

            # Print result
            status = "OK" if is_consistent else "WARN"
            pred_summary = ", ".join([f"{p['rotation']}°:{p['class_name'][:15]}({p['confidence']:.1f}%)"
                                      for p in predictions])
            print(f"    {img_name}: {pred_summary} {'✓' if is_consistent else '✗'}")

    return results


def test_server_api(use_tunnel: bool = False) -> dict:
    """Test the server API with rotated images"""
    print_header("Server API Testing")

    base_url = CONFIG["tunnel_url"] if use_tunnel else CONFIG["server_url"]
    api_key = CONFIG["api_key"]

    print_status(f"Testing server: {base_url}", "INFO")

    results = {
        "server_url": base_url,
        "total_tests": 0,
        "successful": 0,
        "rotation_consistent": 0,
        "detailed_results": [],
    }

    # Check server health
    try:
        health_url = f"{base_url}/health"
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print_status("Server is healthy", "OK")
        else:
            print_status(f"Server health check failed: {response.status_code}", "ERROR")
            return results
    except Exception as e:
        print_status(f"Cannot connect to server: {e}", "ERROR")
        return results

    # Test with porcupine damage images (our problem case)
    test_dir = os.path.join(CONFIG["test_images_dir"], "porcupine   damage")

    if not os.path.exists(test_dir):
        print_status(f"Test directory not found: {test_dir}", "WARN")
        return results

    images = [f for f in os.listdir(test_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]

    headers = {"X-API-Key": api_key}

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        predictions = []

        print_status(f"\nTesting: {img_name}", "INFO")

        for rotation in CONFIG["test_rotations"]:
            # Load, rotate, and send image
            image = Image.open(img_path).convert("RGB")
            if rotation != 0:
                image = image.rotate(-rotation, expand=True)

            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name, "JPEG")
                tmp_path = tmp.name

            try:
                with open(tmp_path, "rb") as f:
                    files = {"file": (f"test_{rotation}.jpg", f, "image/jpeg")}
                    response = requests.post(
                        f"{base_url}/predict",
                        files=files,
                        headers=headers,
                        timeout=30,
                    )

                if response.status_code == 200:
                    data = response.json()
                    pred = {
                        "rotation": rotation,
                        "class_name": data.get("predicted_class", "unknown"),
                        "confidence": data.get("confidence", 0),
                    }
                    predictions.append(pred)
                    results["successful"] += 1
                    print(f"    {rotation}°: {pred['class_name']} ({pred['confidence']:.1f}%)")
                else:
                    print_status(f"    {rotation}°: API error {response.status_code}", "ERROR")
                    predictions.append({"rotation": rotation, "error": response.status_code})

            finally:
                os.unlink(tmp_path)

            results["total_tests"] += 1

        # Check consistency
        valid_preds = [p for p in predictions if "class_name" in p]
        if valid_preds:
            unique_predictions = set(p["class_name"] for p in valid_preds)
            is_consistent = len(unique_predictions) == 1

            if is_consistent:
                results["rotation_consistent"] += 1

            results["detailed_results"].append({
                "image": img_name,
                "predictions": predictions,
                "rotation_consistent": is_consistent,
            })

            status = "OK" if is_consistent else "WARN"
            print_status(f"Rotation consistency: {'Yes' if is_consistent else 'No'}", status)

    return results


def generate_report(backup_path: str, local_results: dict, server_results: dict) -> str:
    """Generate a deployment report"""
    print_header("Deployment Report")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
================================================================================
ROTATION-ROBUST MODEL DEPLOYMENT REPORT
Generated: {timestamp}
================================================================================

1. MODEL DEPLOYMENT
-------------------
New Model: {CONFIG["new_model_path"]}
Deployed To: {CONFIG["current_model_path"]}
Backup Created: {backup_path or "N/A"}

2. LOCAL TESTING RESULTS
------------------------
Total Tests: {local_results.get("total_tests", 0)}
Rotation Consistent: {local_results.get("rotation_consistent", 0)}
Consistency Rate: {local_results.get("rotation_consistent", 0) / max(local_results.get("total_tests", 1), 1) * 100:.1f}%

By Class:
"""

    for class_name, data in local_results.get("by_class", {}).items():
        consistency_rate = data["rotation_consistent"] / max(data["total"], 1) * 100
        report += f"  - {class_name}: {data['rotation_consistent']}/{data['total']} ({consistency_rate:.1f}%)\n"

    report += f"""
3. SERVER API TESTING RESULTS
-----------------------------
Server URL: {server_results.get("server_url", "N/A")}
Total API Calls: {server_results.get("total_tests", 0)}
Successful: {server_results.get("successful", 0)}
Rotation Consistent: {server_results.get("rotation_consistent", 0)}

4. RECOMMENDATIONS
------------------
"""

    local_consistency = local_results.get("rotation_consistent", 0) / max(local_results.get("total_tests", 1), 1)

    if local_consistency >= 0.9:
        report += "✅ Model shows excellent rotation invariance (>90%)\n"
        report += "✅ Ready for production deployment\n"
    elif local_consistency >= 0.7:
        report += "⚠️ Model shows good rotation invariance (70-90%)\n"
        report += "⚠️ Consider additional training if needed\n"
    else:
        report += "❌ Model shows poor rotation invariance (<70%)\n"
        report += "❌ May need more training or architectural changes\n"

    report += """
================================================================================
END OF REPORT
================================================================================
"""

    # Save report
    report_path = os.path.join(
        os.path.dirname(CONFIG["new_model_path"]),
        f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print_status(f"Report saved to: {report_path}", "OK")

    return report_path


# ============================================================
# Main Script
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Deploy and test rotation-robust model")
    parser.add_argument("--force", action="store_true", help="Force deployment even if training not complete")
    parser.add_argument("--skip-tests", action="store_true", help="Skip testing after deployment")
    parser.add_argument("--test-server", action="store_true", help="Also test via server API")
    parser.add_argument("--use-tunnel", action="store_true", help="Use tunnel URL for server tests")
    parser.add_argument("--wait", action="store_true", help="Wait for training to complete")
    args = parser.parse_args()

    print_header("Rotation-Robust Model Deployment Script")
    print_status(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

    # Step 1: Check training status
    print_header("Step 1: Checking Training Status")

    if args.wait:
        print_status("Waiting for training to complete...", "WAIT")
        while True:
            status = check_training_complete()
            if status["ready"]:
                break
            print_status(f"Model not ready yet. Checking again in 60 seconds...", "WAIT")
            time.sleep(60)

    status = check_training_complete()

    if status["exists"]:
        print_status(f"New model found: {status['path']}", "OK")
        print_status(f"Size: {status['size_mb']:.2f} MB", "INFO")
        print_status(f"Modified: {status['modified_time']}", "INFO")
    else:
        if args.force:
            print_status("Model not found but --force specified", "WARN")
        else:
            print_status("New model not found. Training may not be complete.", "ERROR")
            print_status("Use --wait to wait for training or --force to skip check", "INFO")
            return 1

    # Step 2: Backup current model
    print_header("Step 2: Backing Up Current Model")
    backup_path = backup_current_model()

    # Step 3: Deploy new model
    print_header("Step 3: Deploying New Model")
    if not deploy_new_model():
        print_status("Deployment failed!", "ERROR")
        return 1

    # Step 4: Run tests
    local_results = {}
    server_results = {}

    if not args.skip_tests:
        # Local testing
        local_results = test_model_locally(CONFIG["current_model_path"])

        # Server testing (optional)
        if args.test_server:
            print_status("\nNote: Restart the server to load the new model before testing!", "WARN")
            input("Press Enter after restarting the server...")
            server_results = test_server_api(use_tunnel=args.use_tunnel)

    # Step 5: Generate report
    report_path = generate_report(backup_path, local_results, server_results)

    print_header("Deployment Complete!")
    print_status(f"New model deployed to: {CONFIG['current_model_path']}", "OK")
    print_status(f"Backup saved to: {backup_path}", "OK")
    print_status("Remember to restart the server to use the new model!", "WARN")

    return 0


if __name__ == "__main__":
    sys.exit(main())
