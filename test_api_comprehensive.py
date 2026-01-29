"""
Comprehensive API Testing Suite for Intelli-PEST
=================================================
Features:
1. Rotation robustness testing (0°, 90°, 180°, 270°)
2. Checkpoint/Resume - Save progress and continue later
3. Stress testing for tunnel reliability
4. Per-class accuracy breakdown
5. Error analysis by class
6. Multiple test modes

Usage:
    python test_api_comprehensive.py --mode quick       # Quick test (5 images/class)
    python test_api_comprehensive.py --mode full        # Full dataset test
    python test_api_comprehensive.py --mode stress      # Stress test tunnel
    python test_api_comprehensive.py --mode resume      # Resume from checkpoint
    python test_api_comprehensive.py --mode app-test    # Light test for app validation
"""
import argparse
import json
import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import io

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from enhanced_student_model import EnhancedStudentModel

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "LOCAL_API_URL": "http://localhost:8000/api/v1/predict",
    "TUNNEL_API_URL": "https://cnbrr7xn-8000.inc1.devtunnels.ms/api/v1/predict",
    "HEALTH_LOCAL": "http://localhost:8000/api/v1/health",
    "HEALTH_TUNNEL": "https://cnbrr7xn-8000.inc1.devtunnels.ms/api/v1/health",
    "API_KEY": "ip_test_key_intelli_pest_2025",
    "MODEL_PATH": "D:/KnowledgeDistillation/student_model_rotation_robust.pt",
    "TRAINING_DATASET": "G:/AI work/IMAGE DATASET",
    "TEST_IMAGES_DIR": "D:/Test-images",
    "CHECKPOINT_FILE": "D:/KnowledgeDistillation/test_checkpoint.json",
    "RESULTS_FILE": "D:/KnowledgeDistillation/test_results.json",
    "API_TIMEOUT": 90,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

CLASS_NAMES = [
    "Healthy", "Internode borer", "Pink borer", "Rat damage",
    "Stalk borer", "Top borer", "army worm", "mealy bug",
    "porcupine damage", "root borer", "termite"
]

# Image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class TestResult:
    image_path: str
    expected_class: str
    angle: int
    direct_prediction: str
    direct_confidence: float
    direct_correct: bool
    local_prediction: str = ""
    local_confidence: float = 0.0
    local_correct: bool = False
    local_error: str = ""
    tunnel_prediction: str = ""
    tunnel_confidence: float = 0.0
    tunnel_correct: bool = False
    tunnel_error: str = ""
    timestamp: str = ""

@dataclass
class CheckpointData:
    completed_images: List[str]
    results: List[dict]
    start_time: str
    last_update: str
    mode: str
    total_images: int
    
# =============================================================================
# UTILITY CLASSES
# =============================================================================
class GracefulKiller:
    """Handle Ctrl+C gracefully to save checkpoint."""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, *args):
        print("\n\n⚠️  Interrupt received! Saving checkpoint...")
        self.kill_now = True

class SessionManager:
    """Manage HTTP sessions with connection pooling."""
    
    def __init__(self, pool_size: int = 10, retries: int = 3):
        self.local_session = self._create_session(pool_size, retries)
        self.tunnel_session = self._create_session(pool_size, retries)
    
    def _create_session(self, pool_size: int, retries: int) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_size,
            pool_maxsize=pool_size * 2,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def check_health(self, use_tunnel: bool = False) -> Tuple[bool, float]:
        """Check API health and return (is_healthy, response_time)."""
        url = CONFIG["HEALTH_TUNNEL"] if use_tunnel else CONFIG["HEALTH_LOCAL"]
        session = self.tunnel_session if use_tunnel else self.local_session
        
        try:
            start = time.time()
            r = session.get(url, timeout=15, headers={"X-API-Key": CONFIG["API_KEY"]})
            elapsed = time.time() - start
            return r.status_code == 200, elapsed
        except Exception as e:
            return False, 0.0

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model():
    """Load the rotation-robust model."""
    print(f"📦 Loading model from: {CONFIG['MODEL_PATH']}")
    
    checkpoint = torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"])
    print(f"   Upright accuracy: {checkpoint.get('upright_accuracy', 'N/A')}")
    print(f"   Rotation accuracy: {checkpoint.get('rotation_accuracy', 'N/A')}")
    
    model = EnhancedStudentModel(num_classes=11, input_channels=3, base_channels=48)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG["DEVICE"])
    model.eval()
    return model

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def predict_direct(model, img: Image.Image) -> Tuple[str, float]:
    """Direct model inference."""
    tensor = transform(img).unsqueeze(0).to(CONFIG["DEVICE"])
    
    with torch.no_grad():
        output = model(tensor)
        logits = output['logits'] if isinstance(output, dict) else output
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    
    return CLASS_NAMES[pred.item()], conf.item()

def predict_api(img: Image.Image, session: requests.Session, api_url: str, 
                max_retries: int = 3, rate_limit_delay: float = 0.0) -> Tuple[str, float, str]:
    """API prediction with retries."""
    headers = {'X-API-Key': CONFIG["API_KEY"]}
    
    last_error = ""
    for attempt in range(max_retries):
        if rate_limit_delay > 0 and attempt > 0:
            time.sleep(rate_limit_delay)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        files = {'image': ('test.jpg', buffer, 'image/jpeg')}
        
        try:
            response = session.post(api_url, files=files, headers=headers, 
                                   timeout=CONFIG["API_TIMEOUT"])
            if response.status_code == 200:
                data = response.json()
                return data['prediction']['class'], data['prediction']['confidence'], ""
            else:
                try:
                    err = response.json().get('detail', f"HTTP{response.status_code}")
                    last_error = str(err)[:50]
                except:
                    last_error = f"HTTP{response.status_code}"
                if 400 <= response.status_code < 500:
                    break
        except Exception as e:
            last_error = str(e)[:50]
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return "ERR", 0.0, last_error

# =============================================================================
# IMAGE COLLECTION
# =============================================================================
def collect_images(mode: str = "quick", images_per_class: int = 5) -> List[Tuple[str, str, str]]:
    """Collect test images based on mode."""
    test_cases = []
    
    if mode == "app-test":
        # Just a few images for quick app validation
        images_per_class = 2
    elif mode == "quick":
        images_per_class = 5
    elif mode == "full":
        images_per_class = None  # All images
    
    # Training dataset
    dataset_path = Path(CONFIG["TRAINING_DATASET"])
    if dataset_path.exists():
        print(f"\n📁 Collecting from training dataset...")
        for class_dir in sorted(dataset_path.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
                if images:
                    if images_per_class:
                        step = max(1, len(images) // images_per_class)
                        selected = images[::step][:images_per_class]
                    else:
                        selected = images
                    
                    for img in selected:
                        test_cases.append((str(img), class_dir.name, "training"))
                    print(f"   ✓ {class_dir.name}: {len(selected)} images")
    
    # Test images directory
    test_path = Path(CONFIG["TEST_IMAGES_DIR"])
    if test_path.exists() and mode != "app-test":
        print(f"\n📁 Collecting from test directory...")
        for class_dir in sorted(test_path.iterdir()):
            if class_dir.is_dir():
                class_name = ' '.join(class_dir.name.split())
                total = 0
                
                for subfolder in class_dir.iterdir():
                    if subfolder.is_dir():
                        images = list(subfolder.glob("*.JPG")) + list(subfolder.glob("*.jpg"))
                        for img in images:
                            test_cases.append((str(img), class_name, f"test/{subfolder.name}"))
                            total += 1
                
                direct_images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
                for img in direct_images:
                    test_cases.append((str(img), class_name, "test/direct"))
                    total += 1
                
                if total > 0:
                    print(f"   ✓ {class_name}: {total} images")
    
    return test_cases

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================
def save_checkpoint(completed: List[str], results: List[TestResult], mode: str, total: int):
    """Save checkpoint to file."""
    data = CheckpointData(
        completed_images=completed,
        results=[asdict(r) for r in results],
        start_time=datetime.now().isoformat(),
        last_update=datetime.now().isoformat(),
        mode=mode,
        total_images=total
    )
    
    with open(CONFIG["CHECKPOINT_FILE"], 'w') as f:
        json.dump(asdict(data), f, indent=2)
    print(f"💾 Checkpoint saved: {len(completed)}/{total} images completed")

def load_checkpoint() -> Optional[CheckpointData]:
    """Load checkpoint if exists."""
    if not os.path.exists(CONFIG["CHECKPOINT_FILE"]):
        return None
    
    try:
        with open(CONFIG["CHECKPOINT_FILE"], 'r') as f:
            data = json.load(f)
        print(f"📂 Checkpoint found: {len(data['completed_images'])}/{data['total_images']} completed")
        print(f"   Last update: {data['last_update']}")
        return CheckpointData(**data)
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return None

def clear_checkpoint():
    """Clear checkpoint file."""
    if os.path.exists(CONFIG["CHECKPOINT_FILE"]):
        os.remove(CONFIG["CHECKPOINT_FILE"])
        print("🗑️  Checkpoint cleared")

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
def analyze_results(results: List[TestResult]) -> Dict:
    """Analyze test results by class."""
    analysis = {
        "total_tests": len(results),
        "by_class": defaultdict(lambda: {
            "total": 0, "direct_correct": 0, "local_correct": 0, "tunnel_correct": 0,
            "direct_errors": [], "local_errors": [], "tunnel_errors": []
        }),
        "by_angle": defaultdict(lambda: {"total": 0, "direct_correct": 0}),
        "error_summary": {"local": defaultdict(int), "tunnel": defaultdict(int)},
        "overall": {
            "direct_accuracy": 0.0,
            "local_accuracy": 0.0,
            "tunnel_accuracy": 0.0,
        }
    }
    
    for r in results:
        cls = r.expected_class
        analysis["by_class"][cls]["total"] += 1
        analysis["by_angle"][r.angle]["total"] += 1
        
        if r.direct_correct:
            analysis["by_class"][cls]["direct_correct"] += 1
            analysis["by_angle"][r.angle]["direct_correct"] += 1
        else:
            analysis["by_class"][cls]["direct_errors"].append({
                "predicted": r.direct_prediction,
                "image": Path(r.image_path).name,
                "angle": r.angle
            })
        
        if r.local_prediction:
            if r.local_correct:
                analysis["by_class"][cls]["local_correct"] += 1
            if r.local_error:
                analysis["error_summary"]["local"][r.local_error] += 1
        
        if r.tunnel_prediction:
            if r.tunnel_correct:
                analysis["by_class"][cls]["tunnel_correct"] += 1
            if r.tunnel_error:
                analysis["error_summary"]["tunnel"][r.tunnel_error] += 1
    
    # Calculate overall accuracy
    total = len(results)
    if total > 0:
        analysis["overall"]["direct_accuracy"] = sum(1 for r in results if r.direct_correct) / total
        local_tested = [r for r in results if r.local_prediction]
        tunnel_tested = [r for r in results if r.tunnel_prediction]
        
        if local_tested:
            analysis["overall"]["local_accuracy"] = sum(1 for r in local_tested if r.local_correct) / len(local_tested)
        if tunnel_tested:
            analysis["overall"]["tunnel_accuracy"] = sum(1 for r in tunnel_tested if r.tunnel_correct) / len(tunnel_tested)
    
    return analysis

def print_analysis(analysis: Dict):
    """Print detailed analysis."""
    print("\n" + "=" * 90)
    print("📊 DETAILED ANALYSIS")
    print("=" * 90)
    
    # Overall accuracy
    print(f"\n{'Method':<20} | {'Accuracy':>10}")
    print("-" * 35)
    print(f"{'Direct Model':<20} | {analysis['overall']['direct_accuracy']:>9.1%}")
    if analysis['overall']['local_accuracy'] > 0:
        print(f"{'Local API':<20} | {analysis['overall']['local_accuracy']:>9.1%}")
    if analysis['overall']['tunnel_accuracy'] > 0:
        print(f"{'Tunnel API':<20} | {analysis['overall']['tunnel_accuracy']:>9.1%}")
    
    # Per-class breakdown
    print(f"\n{'Class':<20} | {'Total':>6} | {'Direct':>8} | {'Local':>8} | {'Tunnel':>8}")
    print("-" * 65)
    
    for cls in sorted(analysis["by_class"].keys()):
        data = analysis["by_class"][cls]
        total = data["total"]
        direct_acc = data["direct_correct"] / total * 100 if total > 0 else 0
        local_acc = data["local_correct"] / total * 100 if total > 0 else 0
        tunnel_acc = data["tunnel_correct"] / total * 100 if total > 0 else 0
        
        print(f"{cls[:20]:<20} | {total:>6} | {direct_acc:>7.1f}% | {local_acc:>7.1f}% | {tunnel_acc:>7.1f}%")
    
    # Problem classes (below 80% accuracy)
    print("\n⚠️  Problem Classes (Direct accuracy < 80%):")
    for cls in sorted(analysis["by_class"].keys()):
        data = analysis["by_class"][cls]
        total = data["total"]
        if total > 0:
            acc = data["direct_correct"] / total
            if acc < 0.8:
                print(f"   • {cls}: {acc:.1%} ({data['direct_correct']}/{total})")
                # Show common misclassifications
                errors = data["direct_errors"][:3]
                for e in errors:
                    print(f"     → Predicted '{e['predicted']}' at {e['angle']}° ({e['image']})")
    
    # By rotation angle
    print(f"\n{'Angle':<10} | {'Total':>6} | {'Direct Accuracy':>15}")
    print("-" * 40)
    for angle in [0, 90, 180, 270]:
        data = analysis["by_angle"][angle]
        if data["total"] > 0:
            acc = data["direct_correct"] / data["total"] * 100
            print(f"{angle:>5}°     | {data['total']:>6} | {acc:>14.1f}%")
    
    # API errors
    if analysis["error_summary"]["local"] or analysis["error_summary"]["tunnel"]:
        print("\n⚠️  API Error Summary:")
        if analysis["error_summary"]["local"]:
            print("   Local API:")
            for err, count in sorted(analysis["error_summary"]["local"].items(), key=lambda x: -x[1])[:5]:
                print(f"     • {err}: {count}x")
        if analysis["error_summary"]["tunnel"]:
            print("   Tunnel API:")
            for err, count in sorted(analysis["error_summary"]["tunnel"].items(), key=lambda x: -x[1])[:5]:
                print(f"     • {err}: {count}x")

# =============================================================================
# STRESS TEST
# =============================================================================
def run_stress_test(sessions: SessionManager, duration_seconds: int = 60, 
                    requests_per_second: float = 2.0):
    """Run stress test against the tunnel."""
    print("\n" + "=" * 90)
    print("🔥 STRESS TEST - Tunnel Reliability")
    print("=" * 90)
    print(f"Duration: {duration_seconds}s | Target RPS: {requests_per_second}")
    
    # Check tunnel health first
    healthy, response_time = sessions.check_health(use_tunnel=True)
    if not healthy:
        print("❌ Tunnel not accessible! Cannot run stress test.")
        return
    
    print(f"Initial tunnel response time: {response_time:.2f}s")
    
    # Create a test image
    test_img = Image.new('RGB', (256, 256), color='green')
    
    results = {
        "successful": 0,
        "failed": 0,
        "timeouts": 0,
        "response_times": [],
        "errors": defaultdict(int)
    }
    
    start_time = time.time()
    request_interval = 1.0 / requests_per_second
    next_request_time = start_time
    
    print("\nRunning stress test... (Ctrl+C to stop)")
    
    try:
        while time.time() - start_time < duration_seconds:
            # Wait for next scheduled request
            now = time.time()
            if now < next_request_time:
                time.sleep(next_request_time - now)
            
            # Send request
            req_start = time.time()
            cls, conf, err = predict_api(
                test_img, sessions.tunnel_session, 
                CONFIG["TUNNEL_API_URL"], max_retries=1
            )
            req_time = time.time() - req_start
            
            if err:
                results["failed"] += 1
                results["errors"][err] += 1
                if "timeout" in err.lower():
                    results["timeouts"] += 1
                status = f"❌ {err[:30]}"
            else:
                results["successful"] += 1
                results["response_times"].append(req_time)
                status = f"✅ {req_time:.2f}s"
            
            elapsed = time.time() - start_time
            total = results["successful"] + results["failed"]
            print(f"\r[{elapsed:5.1f}s] Requests: {total} | Success: {results['successful']} | "
                  f"Failed: {results['failed']} | Last: {status}     ", end="")
            
            next_request_time += request_interval
            
    except KeyboardInterrupt:
        print("\n\nStress test interrupted.")
    
    # Print results
    print("\n\n" + "-" * 50)
    print("STRESS TEST RESULTS")
    print("-" * 50)
    total = results["successful"] + results["failed"]
    success_rate = results["successful"] / total * 100 if total > 0 else 0
    
    print(f"Total requests:  {total}")
    print(f"Successful:      {results['successful']} ({success_rate:.1f}%)")
    print(f"Failed:          {results['failed']}")
    print(f"Timeouts:        {results['timeouts']}")
    
    if results["response_times"]:
        avg_time = sum(results["response_times"]) / len(results["response_times"])
        max_time = max(results["response_times"])
        min_time = min(results["response_times"])
        print(f"Response time:   avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
    
    if results["errors"]:
        print("\nErrors:")
        for err, count in sorted(results["errors"].items(), key=lambda x: -x[1]):
            print(f"  • {err}: {count}x")
    
    # Verdict
    print("\n" + "=" * 50)
    if success_rate >= 95:
        print("✅ PASS: Tunnel is stable (>95% success rate)")
    elif success_rate >= 80:
        print("⚠️  WARNING: Tunnel has some issues (80-95% success)")
    else:
        print("❌ FAIL: Tunnel is unstable (<80% success rate)")

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def run_rotation_test(mode: str = "quick", test_local: bool = True, 
                      test_tunnel: bool = False, resume: bool = False,
                      workers: int = 4):
    """Run the main rotation test."""
    killer = GracefulKiller()
    sessions = SessionManager(pool_size=workers)
    
    print("\n" + "=" * 90)
    print(f"🧪 ROTATION ROBUSTNESS TEST - Mode: {mode.upper()}")
    print("=" * 90)
    
    # Check API availability
    print("\n🔍 Checking API connectivity...")
    local_ok, local_time = sessions.check_health(use_tunnel=False)
    print(f"   Local API:  {'✅ Available' if local_ok else '❌ Not available'} ({local_time:.2f}s)")
    
    if test_tunnel:
        tunnel_ok, tunnel_time = sessions.check_health(use_tunnel=True)
        print(f"   Tunnel API: {'✅ Available' if tunnel_ok else '❌ Not available'} ({tunnel_time:.2f}s)")
        if not tunnel_ok:
            print("   ⚠️  Skipping tunnel tests")
            test_tunnel = False
    
    if not local_ok and not test_tunnel:
        print("\n❌ No API available! Start server with: python run_server.py")
        return
    
    # Load model
    model = load_model()
    
    # Collect images
    test_cases = collect_images(mode)
    print(f"\n📊 Total images to test: {len(test_cases)}")
    
    # Check for checkpoint
    completed_paths = set()
    results = []
    
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_paths = set(checkpoint.completed_images)
            results = [TestResult(**r) for r in checkpoint.results]
            print(f"   Resuming from checkpoint: {len(completed_paths)} already done")
    
    # Filter out completed
    remaining = [(p, c, s) for p, c, s in test_cases if p not in completed_paths]
    print(f"   Remaining to test: {len(remaining)}")
    
    if not remaining:
        print("\n✅ All images already tested!")
        if results:
            analysis = analyze_results(results)
            print_analysis(analysis)
        return
    
    # Run tests
    start_time = time.time()
    checkpoint_interval = 50  # Save every 50 images
    
    for idx, (path, expected_class, source) in enumerate(remaining):
        if killer.kill_now:
            save_checkpoint(list(completed_paths), results, mode, len(test_cases))
            print("\n⏸️  Test paused. Run with --mode resume to continue.")
            return
        
        if not Path(path).exists():
            continue
        
        # Load and rotate image
        img = Image.open(path).convert('RGB')
        
        for angle in [0, 90, 180, 270]:
            rotated = img.rotate(-angle, expand=True)
            
            # Direct model prediction
            direct_cls, direct_conf = predict_direct(model, rotated)
            direct_ok = direct_cls.lower() == expected_class.lower()
            
            result = TestResult(
                image_path=path,
                expected_class=expected_class,
                angle=angle,
                direct_prediction=direct_cls,
                direct_confidence=direct_conf,
                direct_correct=direct_ok,
                timestamp=datetime.now().isoformat()
            )
            
            # Local API (if enabled)
            if test_local and local_ok:
                local_cls, local_conf, local_err = predict_api(
                    rotated, sessions.local_session, CONFIG["LOCAL_API_URL"], max_retries=2
                )
                result.local_prediction = local_cls
                result.local_confidence = local_conf
                result.local_correct = local_cls.lower() == expected_class.lower()
                result.local_error = local_err
            
            # Tunnel API (if enabled)
            if test_tunnel:
                tunnel_cls, tunnel_conf, tunnel_err = predict_api(
                    rotated, sessions.tunnel_session, CONFIG["TUNNEL_API_URL"], 
                    max_retries=3, rate_limit_delay=0.5
                )
                result.tunnel_prediction = tunnel_cls
                result.tunnel_confidence = tunnel_conf
                result.tunnel_correct = tunnel_cls.lower() == expected_class.lower()
                result.tunnel_error = tunnel_err
            
            results.append(result)
        
        completed_paths.add(path)
        
        # Progress update
        progress = (idx + 1) / len(remaining) * 100
        elapsed = time.time() - start_time
        eta = elapsed / (idx + 1) * (len(remaining) - idx - 1) if idx > 0 else 0
        
        direct_acc = sum(1 for r in results if r.direct_correct) / len(results) * 100
        
        print(f"\r[{progress:5.1f}%] {idx + 1}/{len(remaining)} | "
              f"Direct: {direct_acc:.1f}% | "
              f"ETA: {eta/60:.1f}min     ", end="")
        
        # Periodic checkpoint
        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(list(completed_paths), results, mode, len(test_cases))
    
    # Final save
    save_checkpoint(list(completed_paths), results, mode, len(test_cases))
    
    # Analysis
    print("\n")
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    # Save full results
    with open(CONFIG["RESULTS_FILE"], 'w') as f:
        json.dump({
            "analysis": {k: dict(v) if isinstance(v, defaultdict) else v 
                        for k, v in analysis.items()},
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "total_results": len(results)
        }, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {CONFIG['RESULTS_FILE']}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Comprehensive API Testing Suite")
    parser.add_argument("--mode", choices=["quick", "full", "stress", "resume", "app-test"],
                       default="quick", help="Test mode")
    parser.add_argument("--tunnel", action="store_true", help="Include tunnel testing")
    parser.add_argument("--no-local", action="store_true", help="Skip local API testing")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--stress-duration", type=int, default=60, help="Stress test duration (seconds)")
    parser.add_argument("--stress-rps", type=float, default=2.0, help="Stress test requests per second")
    parser.add_argument("--clear-checkpoint", action="store_true", help="Clear existing checkpoint")
    
    args = parser.parse_args()
    
    print("=" * 90)
    print("🔬 INTELLI-PEST COMPREHENSIVE API TEST SUITE")
    print("=" * 90)
    print(f"Mode: {args.mode} | Tunnel: {args.tunnel} | Workers: {args.workers}")
    
    if args.clear_checkpoint:
        clear_checkpoint()
    
    if args.mode == "stress":
        sessions = SessionManager()
        run_stress_test(sessions, args.stress_duration, args.stress_rps)
    else:
        run_rotation_test(
            mode=args.mode if args.mode != "resume" else "quick",
            test_local=not args.no_local,
            test_tunnel=args.tunnel,
            resume=args.mode == "resume",
            workers=args.workers
        )

if __name__ == "__main__":
    main()
