"""
Test rotation robustness via API, Direct Model, AND Application Tunnel
=======================================================================
Send the same image at 0°, 90°, 180°, 270° rotations and compare predictions.
Compare results from:
1. Direct model inference (local)
2. Local API server (localhost:8000)
3. Application tunnel (VS Code Dev Tunnels - what the Android app uses)

This gives complete end-to-end validation.

OPTIMIZATIONS:
- Parallel API calls using ThreadPoolExecutor
- Connection pooling with requests.Session
- Batch processing of rotations
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from pathlib import Path
import io
import sys
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add src to path for model loading
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from enhanced_student_model import EnhancedStudentModel

# Configuration
LOCAL_API_URL = "http://localhost:8000/api/v1/predict"
TUNNEL_API_URL = "https://cnbrr7xn-8000.inc1.devtunnels.ms/api/v1/predict"  # VS Code Dev Tunnel
API_KEY = "ip_test_key_intelli_pest_2025"
MODEL_PATH = "D:/KnowledgeDistillation/student_model_rotation_robust.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_TIMEOUT = 90  # seconds

# Test image sources
TRAINING_DATASET = "G:/AI work/IMAGE DATASET"
TEST_IMAGES_DIR = "D:/Test-images"

CLASS_NAMES = [
    "Healthy", "Internode borer", "Pink borer", "Rat damage", 
    "Stalk borer", "Top borer", "army worm", "mealy bug", 
    "porcupine damage", "root borer", "termite"
]

# Standard transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create session with connection pooling and retries
def create_session():
    """Create a requests session with connection pooling and retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Global sessions for connection reuse
local_session = create_session()
tunnel_session = create_session()


def load_model():
    """Load the rotation-robust model directly."""
    print(f"Loading model from: {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  Upright accuracy: {checkpoint.get('upright_accuracy', 'N/A')}")
    print(f"  Rotation accuracy: {checkpoint.get('rotation_accuracy', 'N/A')}")
    
    model = EnhancedStudentModel(
        num_classes=11,
        input_channels=3,
        base_channels=48
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def predict_direct(model, img: Image.Image):
    """Predict using direct model inference."""
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        logits = output['logits'] if isinstance(output, dict) else output
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        
    return CLASS_NAMES[pred.item()], conf.item()


def predict_api(img: Image.Image, api_url: str = LOCAL_API_URL, session=None, max_retries: int = 1):
    """Predict using API endpoint (local or tunnel).
    
    Args:
        img: PIL Image to classify
        api_url: API endpoint URL
        session: requests.Session for connection pooling
        max_retries: Number of retry attempts (default 1, use 3 for tunnel)
    """
    headers = {'X-API-Key': API_KEY}
    
    # Use session for connection pooling, fall back to requests if no session
    req = session if session else requests
    
    last_error = None
    for attempt in range(max_retries):
        # Create fresh buffer for each attempt
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        files = {'image': ('test.jpg', buffer, 'image/jpeg')}
        
        try:
            response = req.post(api_url, files=files, headers=headers, timeout=API_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                return data['prediction']['class'], data['prediction']['confidence'], None
            else:
                # Extract error detail if available
                try:
                    err_detail = response.json().get('detail', {})
                    if isinstance(err_detail, dict):
                        err_code = err_detail.get('code', 'UNKNOWN')
                    else:
                        err_code = str(err_detail)[:30]
                except:
                    err_code = f"HTTP{response.status_code}"
                last_error = err_code
                # Don't retry on validation errors (4xx), only on server/network errors
                if 400 <= response.status_code < 500:
                    break
        except Exception as e:
            last_error = str(e)[:30]
            # Continue to retry on network errors
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
                continue
    
    return "ERR", 0.0, last_error


def test_image_rotations(image_path: str, expected_class: str, model=None, test_local: bool = True, test_tunnel: bool = True):
    """Test an image at all 4 rotations - Direct model, Local API, and Tunnel API.
    Uses parallel API calls for speed."""
    print(f"\n{'='*90}")
    print(f"Testing: {Path(image_path).name}")
    print(f"Expected: {expected_class}")
    print(f"{'='*90}")
    
    # Build header based on what we're testing
    cols = ["DIRECT MODEL"]
    if test_local:
        cols.append("LOCAL API")
    if test_tunnel:
        cols.append("TUNNEL API")
    
    header = f"{'Angle':>6}"
    for col in cols:
        header += f" | {col:^22}"
    print(header)
    print(f"{'-'*6}" + "-+-".join(["-"*22 for _ in cols]))
    
    img = Image.open(image_path).convert('RGB')
    
    # Pre-rotate all images
    rotated_images = {angle: img.rotate(-angle, expand=True) for angle in [0, 90, 180, 270]}
    
    # Batch direct model predictions (fast, no parallelization needed)
    direct_results = {}
    if model is not None:
        for angle, rotated in rotated_images.items():
            direct_results[angle] = predict_direct(model, rotated)
    
    # Parallel API calls for Local and Tunnel
    api_results = {}
    errors = []
    
    def call_api(angle, rotated, api_url, api_name, session, retries=1):
        cls, conf, err = predict_api(rotated, api_url, session, max_retries=retries)
        return (angle, api_name, cls, conf, err)
    
    # Submit all API calls in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for angle, rotated in rotated_images.items():
            if test_local:
                futures.append(executor.submit(call_api, angle, rotated, LOCAL_API_URL, "local", local_session, 1))
            if test_tunnel:
                # Use 3 retries for tunnel API due to network instability
                futures.append(executor.submit(call_api, angle, rotated, TUNNEL_API_URL, "tunnel", tunnel_session, 3))
        
        # Collect results
        for future in as_completed(futures):
            angle, api_name, cls, conf, err = future.result()
            api_results[(angle, api_name)] = (cls, conf)
            if err:
                errors.append(f"{api_name.title()}[{angle}°]: {err}")
    
    # Format and print results
    results = []
    for angle in [0, 90, 180, 270]:
        # Direct model result
        if model is not None:
            direct_class, direct_conf = direct_results[angle]
            direct_ok = "✅" if direct_class.lower() == expected_class.lower() else "❌"
            direct_str = f"{direct_class[:12]:12} {direct_conf:5.1%} {direct_ok}"
        else:
            direct_str = "N/A"
            direct_ok = "?"
        
        # Local API result
        local_ok = "?"
        local_str = "SKIP"
        if test_local and (angle, "local") in api_results:
            local_class, local_conf = api_results[(angle, "local")]
            local_ok = "✅" if local_class.lower() == expected_class.lower() else "❌"
            local_str = f"{local_class[:12]:12} {local_conf:5.1%} {local_ok}"
        
        # Tunnel API result
        tunnel_ok = "?"
        tunnel_str = "SKIP"
        if test_tunnel and (angle, "tunnel") in api_results:
            tunnel_class, tunnel_conf = api_results[(angle, "tunnel")]
            tunnel_ok = "✅" if tunnel_class.lower() == expected_class.lower() else "❌"
            tunnel_str = f"{tunnel_class[:12]:12} {tunnel_conf:5.1%} {tunnel_ok}"
        
        # Build output line
        line = f"{angle:>5}° | {direct_str:^22}"
        if test_local:
            line += f" | {local_str:^22}"
        if test_tunnel:
            line += f" | {tunnel_str:^22}"
        print(line)
        
        results.append((angle, direct_ok, local_ok, tunnel_ok))
    
    if errors:
        print(f"  ⚠️ Errors: {', '.join(set(errors))}")
    
    return results


def collect_test_images(images_per_class: int = None):
    """Collect ALL test images from both training dataset and D:/Test-images.
    
    Args:
        images_per_class: If None, collect ALL images. If specified, limit per class.
    """
    test_cases = []
    
    # 1. From training dataset (G:/AI work/IMAGE DATASET) - ALL images or limited
    if images_per_class:
        print(f"\n📁 Collecting from training dataset ({images_per_class} per class)...")
    else:
        print(f"\n📁 Collecting ALL images from training dataset...")
    
    dataset_path = Path(TRAINING_DATASET)
    if dataset_path.exists():
        for class_dir in sorted(dataset_path.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
                
                if images:
                    if images_per_class:
                        # Limit to images_per_class, spread across the folder
                        step = max(1, len(images) // images_per_class)
                        selected = images[::step][:images_per_class]
                    else:
                        # Take ALL images
                        selected = images
                    
                    for img in selected:
                        test_cases.append((str(img), class_dir.name, "training"))
                    print(f"   ✓ {class_dir.name}: {len(selected)} images")
    
    # 2. From D:/Test-images (ALL real-world test images with subfolders)
    print(f"\n📁 Collecting ALL images from {TEST_IMAGES_DIR}...")
    test_path = Path(TEST_IMAGES_DIR)
    if test_path.exists():
        for class_dir in sorted(test_path.iterdir()):
            if class_dir.is_dir():
                # Get class name from folder (handle spaces in name like "porcupine   damage")
                class_name = class_dir.name.strip()
                # Normalize class name (remove extra spaces)
                class_name = ' '.join(class_name.split())
                
                total_images = 0
                
                # Look for images in subfolders (Controlled leaf, Real time leaf, etc.)
                for subfolder in class_dir.iterdir():
                    if subfolder.is_dir():
                        images = list(subfolder.glob("*.JPG")) + list(subfolder.glob("*.jpg"))
                        for img in images:
                            test_cases.append((str(img), class_name, f"test/{subfolder.name}"))
                            total_images += 1
                
                # Also check for images directly in class folder
                direct_images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
                for img in direct_images:
                    test_cases.append((str(img), class_name, "test/direct"))
                    total_images += 1
                
                if total_images > 0:
                    print(f"   ✓ {class_name}: {total_images} images")
    
    return test_cases


if __name__ == "__main__":
    start_time = time.time()
    
    print("="*90)
    print("COMPLETE ROTATION ROBUSTNESS TEST - ALL IMAGES (OPTIMIZED)")
    print("Direct Model vs Local API vs Tunnel API (Android App)")
    print("="*90)
    print(f"Local API:  {LOCAL_API_URL}")
    print(f"Tunnel API: {TUNNEL_API_URL}")
    print(f"Model:      {MODEL_PATH}")
    print(f"Timeout:    {API_TIMEOUT}s | Tunnel retries: 3 | Parallel workers: 12")
    print()
    
    # Check if local API is accessible
    print("🔍 Checking API connectivity...")
    local_health_url = "http://localhost:8000/api/v1/health"
    try:
        r = local_session.get(local_health_url, timeout=10)
        local_available = r.status_code == 200
        print(f"   Local API:  {'✅ Available' if local_available else '❌ Not available (status ' + str(r.status_code) + ')'}")
    except Exception as e:
        local_available = False
        print(f"   Local API:  ❌ Not available ({str(e)[:50]})")
    
    # Check if tunnel is accessible
    tunnel_health_url = "https://cnbrr7xn-8000.inc1.devtunnels.ms/api/v1/health"
    try:
        r = tunnel_session.get(tunnel_health_url, timeout=15)
        tunnel_available = r.status_code == 200
        print(f"   Tunnel API: {'✅ Available' if tunnel_available else '❌ Not available (status ' + str(r.status_code) + ')'}")
    except Exception as e:
        tunnel_available = False
        print(f"   Tunnel API: ❌ Not available ({str(e)[:50]})")
    
    if not local_available:
        print("\n⚠️  Local API not running. Start with: python run_server.py")
        print("    Continuing with Direct Model only...")
    
    # Load the model directly for comparison
    print()
    model = load_model()
    
    # Collect all test images
    print()
    test_cases = collect_test_images()
    print(f"\n📊 Total test images: {len(test_cases)}")
    
    # Run tests
    total_direct = 0
    total_local = 0
    total_tunnel = 0
    total_tests = 0
    
    for path, cls, source in test_cases:
        if not Path(path).exists():
            print(f"⚠️ Skipping (not found): {path}")
            continue
            
        results = test_image_rotations(path, cls, model, test_local=local_available, test_tunnel=tunnel_available)
        
        for angle, direct_ok, local_ok, tunnel_ok in results:
            if direct_ok == "✅":
                total_direct += 1
            if local_ok == "✅":
                total_local += 1
            if tunnel_ok == "✅":
                total_tunnel += 1
            total_tests += 1
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"Total images tested: {len(test_cases)}")
    print(f"Total rotations tested: {total_tests}")
    print()
    print(f"{'Method':<20} | {'Correct':>10} | {'Total':>10} | {'Accuracy':>10}")
    print(f"{'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"{'Direct Model':<20} | {total_direct:>10} | {total_tests:>10} | {100*total_direct/total_tests:>9.1f}%")
    if local_available:
        print(f"{'Local API':<20} | {total_local:>10} | {total_tests:>10} | {100*total_local/total_tests:>9.1f}%")
    if tunnel_available:
        print(f"{'Tunnel API (App)':<20} | {total_tunnel:>10} | {total_tests:>10} | {100*total_tunnel/total_tests:>9.1f}%")
    
    print()
    # Check consistency
    all_consistent = True
    if local_available and total_direct != total_local:
        print("⚠️  Local API differs from direct model - check server pipeline")
        all_consistent = False
    if tunnel_available and local_available and total_local != total_tunnel:
        print("⚠️  Tunnel API differs from local - check tunnel/network")
        all_consistent = False
    if tunnel_available and total_direct != total_tunnel:
        print("⚠️  Tunnel API differs from direct model - end-to-end issue")
        all_consistent = False
    
    if all_consistent:
        print("✅ All tested methods produce consistent results!")
    
    # Print elapsed time
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total test time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"   Average per image: {elapsed/len(test_cases):.1f} seconds")
    print("="*90)
