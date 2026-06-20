import time
import threading
import psutil
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

class SystemMonitor:
    def __init__(self):
        self.cpu_usage: List[float] = []
        self.ram_usage: List[float] = []
        self.gpu_usage: List[float] = [] # On Mac this will represent "MPS activity" if available
        self.timestamps: List[float] = []
        self.start_time: float = None
        self._stop: bool = False

    def start(self) -> None:
        self.start_time = time.time()
        self._stop = False
        # daemon=True ensures that the thread terminates along with the main program
        threading.Thread(target=self._monitor, daemon=True).start()

    def _monitor(self) -> None:
        while not self._stop:
            try:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                
                # On Mac M4, we cannot easily retrieve GPU utilization percentage via Python without external utilities,
                # but we can track how much memory is allocated for tensors
                gpu = 0.0
                if torch.backends.mps.is_available():
                    # Placeholder: GPU monitoring on Mac requires `sudo powermetrics`,
                    # so we simply log the presence of active MPS allocation
                    gpu = 1.0 if torch.mps.current_allocated_memory() > 0 else 0.0

                timestamp = (time.time() - self.start_time) / 60

                self.cpu_usage.append(cpu)
                self.ram_usage.append(ram)
                self.gpu_usage.append(gpu)
                self.timestamps.append(timestamp)
            except Exception:
                pass
            time.sleep(2) # Increased interval to prevent the monitoring itself from impacting performance

    def stop(self) -> None:
        self._stop = True

    def get_data(self) -> Dict[str, List[float]]:
        return {
            'timestamps': self.timestamps,
            'cpu': self.cpu_usage,
            'ram': self.ram_usage,
            'gpu': self.gpu_usage
        }

def plot_system_usage(monitor_data, save_dir="plots", algorithm_name="dqn", run_id=None):
    if not monitor_data['timestamps']: return
    
    save_path = Path(save_dir) / algorithm_name.lower()
    save_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(monitor_data['timestamps'], monitor_data['cpu'], label='CPU %', color='blue')
    plt.plot(monitor_data['timestamps'], monitor_data['ram'], label='RAM %', color='orange')
    
    plt.title(f"System Resource Usage ({algorithm_name})")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Usage %")
    plt.legend()
    plt.grid(True)
    
    full_path = save_path / f"sys_usage_{run_id}.png"
    plt.savefig(full_path)
    plt.close()
    print(f"System resource plot saved to: {full_path}")