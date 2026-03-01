import requests
import time
import subprocess
import os
import sys

# Start API Server
print("Starting API Server in background...")
server_process = subprocess.Popen([sys.executable, "d:/AntiGravity/zomathon/src/api/7_api_server.py"])

print("Waiting 15 seconds for server to load mock DB and models...")
time.sleep(15)

url = "http://localhost:8000/recommend"
data = {
    "user_id": 100,
    "restaurant_id": 500,
    "current_cart_item_ids": [1000]
}

print("\n--- Sending 10 Inference Requests ---")
latencies = []

try:
    # First request is warm-up
    r = requests.post(url, json=data)
    print("Warm-up complete.")
    
    for i in range(10):
        start = time.time()
        resp = requests.post(url, json=data)
        net_time = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            res_json = resp.json()
            server_time = res_json['inference_time_ms']
            latencies.append(server_time)
            print(f"Req {i+1}: Recommended {res_json['recommended_item_ids']} | Server CPU Time: {server_time:.2f}ms | Total Net Time (User to Server): {net_time:.2f}ms")
        else:
            print(f"Error {resp.status_code}: {resp.text}")
            
    print(f"\nAverage Server Inference Time: {sum(latencies)/len(latencies):.2f}ms")
    if sum(latencies)/len(latencies) < 300:
        print("SUCCESS: Latency is well under the 300ms required threshold!")
    else:
        print("WARNING: SLA breached.")
        
finally:
    print("Terminating server...")
    server_process.terminate()
