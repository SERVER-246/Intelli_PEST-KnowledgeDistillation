"""
Quick Tunnel Health & Load Monitor for Intelli-PEST App
========================================================
Use this to monitor tunnel status while testing the Android app.

Usage:
    python tunnel_monitor.py              # Continuous monitoring
    python tunnel_monitor.py --once       # Single check
    python tunnel_monitor.py --load-test  # Quick load test (10 requests)
"""
import argparse
import time
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
TUNNEL_URL = "https://cnbrr7xn-8000.inc1.devtunnels.ms"
LOCAL_URL = "http://localhost:8000"
API_KEY = "ip_test_key_intelli_pest_2025"

def create_session():
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=1))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def check_endpoint(session, base_url, name):
    """Check an endpoint and return (success, response_time, status_code)."""
    url = f"{base_url}/api/v1/health"
    headers = {"X-API-Key": API_KEY}
    
    try:
        start = time.time()
        r = session.get(url, timeout=30, headers=headers)
        elapsed = time.time() - start
        return True, elapsed, r.status_code
    except requests.exceptions.Timeout:
        return False, 30.0, "TIMEOUT"
    except requests.exceptions.ConnectionError as e:
        return False, 0, f"CONN_ERR"
    except Exception as e:
        return False, 0, str(e)[:20]

def print_status(timestamp, local_ok, local_time, local_status, 
                tunnel_ok, tunnel_time, tunnel_status):
    """Print status line."""
    local_str = f"✅ {local_time:.2f}s" if local_ok else f"❌ {local_status}"
    tunnel_str = f"✅ {tunnel_time:.2f}s" if tunnel_ok else f"❌ {tunnel_status}"
    
    # Color coding for tunnel response time
    if tunnel_ok:
        if tunnel_time < 1:
            tunnel_quality = "🟢"
        elif tunnel_time < 3:
            tunnel_quality = "🟡"
        elif tunnel_time < 10:
            tunnel_quality = "🟠"
        else:
            tunnel_quality = "🔴"
    else:
        tunnel_quality = "⚫"
    
    print(f"[{timestamp}] Local: {local_str:15} | Tunnel: {tunnel_str:15} {tunnel_quality}")

def continuous_monitor(interval=5):
    """Continuously monitor endpoints."""
    print("=" * 70)
    print("🔍 TUNNEL MONITOR - Watching Server Health")
    print("=" * 70)
    print(f"Local:  {LOCAL_URL}")
    print(f"Tunnel: {TUNNEL_URL}")
    print(f"Interval: {interval}s")
    print("-" * 70)
    print("Response time quality: 🟢<1s | 🟡<3s | 🟠<10s | 🔴>10s | ⚫offline")
    print("-" * 70)
    print("Press Ctrl+C to stop\n")
    
    session = create_session()
    
    stats = {
        "local_success": 0, "local_fail": 0,
        "tunnel_success": 0, "tunnel_fail": 0,
        "tunnel_times": []
    }
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Check both endpoints
            local_ok, local_time, local_status = check_endpoint(session, LOCAL_URL, "local")
            tunnel_ok, tunnel_time, tunnel_status = check_endpoint(session, TUNNEL_URL, "tunnel")
            
            # Update stats
            if local_ok:
                stats["local_success"] += 1
            else:
                stats["local_fail"] += 1
            
            if tunnel_ok:
                stats["tunnel_success"] += 1
                stats["tunnel_times"].append(tunnel_time)
            else:
                stats["tunnel_fail"] += 1
            
            print_status(timestamp, local_ok, local_time, local_status,
                        tunnel_ok, tunnel_time, tunnel_status)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        
        total_local = stats["local_success"] + stats["local_fail"]
        total_tunnel = stats["tunnel_success"] + stats["tunnel_fail"]
        
        print(f"Local API:  {stats['local_success']}/{total_local} successful "
              f"({stats['local_success']/total_local*100:.1f}%)" if total_local > 0 else "No checks")
        
        if total_tunnel > 0:
            print(f"Tunnel API: {stats['tunnel_success']}/{total_tunnel} successful "
                  f"({stats['tunnel_success']/total_tunnel*100:.1f}%)")
            
            if stats["tunnel_times"]:
                avg_time = sum(stats["tunnel_times"]) / len(stats["tunnel_times"])
                max_time = max(stats["tunnel_times"])
                min_time = min(stats["tunnel_times"])
                print(f"Tunnel response: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")

def single_check():
    """Single health check."""
    print("🔍 Quick Health Check")
    print("-" * 50)
    
    session = create_session()
    
    local_ok, local_time, local_status = check_endpoint(session, LOCAL_URL, "local")
    tunnel_ok, tunnel_time, tunnel_status = check_endpoint(session, TUNNEL_URL, "tunnel")
    
    print(f"Local API:  {'✅ OK' if local_ok else '❌ FAIL'} ({local_time:.2f}s) [{local_status}]")
    print(f"Tunnel API: {'✅ OK' if tunnel_ok else '❌ FAIL'} ({tunnel_time:.2f}s) [{tunnel_status}]")
    
    if tunnel_ok:
        if tunnel_time < 1:
            print("\n✅ Tunnel is fast - App should work great!")
        elif tunnel_time < 3:
            print("\n⚠️  Tunnel is moderate - App should work but may be slow")
        elif tunnel_time < 10:
            print("\n⚠️  Tunnel is slow - App may timeout on some requests")
        else:
            print("\n🔴 Tunnel is very slow - App will likely fail")
    else:
        print("\n❌ Tunnel is offline - App cannot connect!")

def load_test(num_requests=10, parallel=2):
    """Quick load test."""
    print("=" * 70)
    print(f"🔥 QUICK LOAD TEST - {num_requests} requests, {parallel} parallel")
    print("=" * 70)
    
    session = create_session()
    
    # First check if tunnel is available
    ok, time_taken, status = check_endpoint(session, TUNNEL_URL, "tunnel")
    if not ok:
        print(f"❌ Tunnel not available: {status}")
        return
    
    print(f"Initial response time: {time_taken:.2f}s\n")
    
    results = {"success": 0, "fail": 0, "times": [], "errors": []}
    start_time = time.time()
    
    def make_request(i):
        ok, elapsed, status = check_endpoint(session, TUNNEL_URL, f"req_{i}")
        return ok, elapsed, status
    
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        
        for i, future in enumerate(as_completed(futures)):
            ok, elapsed, status = future.result()
            if ok:
                results["success"] += 1
                results["times"].append(elapsed)
                print(f"  [{i+1:2}/{num_requests}] ✅ {elapsed:.2f}s")
            else:
                results["fail"] += 1
                results["errors"].append(status)
                print(f"  [{i+1:2}/{num_requests}] ❌ {status}")
    
    total_time = time.time() - start_time
    
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Total time:    {total_time:.2f}s")
    print(f"Success rate:  {results['success']}/{num_requests} ({results['success']/num_requests*100:.1f}%)")
    
    if results["times"]:
        avg = sum(results["times"]) / len(results["times"])
        print(f"Avg response:  {avg:.2f}s")
        print(f"Min response:  {min(results['times']):.2f}s")
        print(f"Max response:  {max(results['times']):.2f}s")
    
    if results["errors"]:
        print(f"\nErrors: {', '.join(set(results['errors']))}")
    
    # Verdict
    success_rate = results["success"] / num_requests
    if success_rate >= 0.9:
        print("\n✅ TUNNEL STABLE - Safe to test app")
    elif success_rate >= 0.7:
        print("\n⚠️  TUNNEL UNSTABLE - App may have intermittent failures")
    else:
        print("\n❌ TUNNEL OVERLOADED - Stop other tests before using app")

def main():
    parser = argparse.ArgumentParser(description="Tunnel Health Monitor")
    parser.add_argument("--once", action="store_true", help="Single check only")
    parser.add_argument("--load-test", action="store_true", help="Quick load test")
    parser.add_argument("--interval", type=int, default=5, help="Monitor interval (seconds)")
    parser.add_argument("--requests", type=int, default=10, help="Load test request count")
    parser.add_argument("--parallel", type=int, default=2, help="Load test parallel requests")
    
    args = parser.parse_args()
    
    if args.once:
        single_check()
    elif args.load_test:
        load_test(args.requests, args.parallel)
    else:
        continuous_monitor(args.interval)

if __name__ == "__main__":
    main()
