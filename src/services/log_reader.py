import os
from typing import Dict, Optional, List
import yaml
from src.utils.parser import LogParser

class LogReader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.root_dir = config['paths']['log_root']
        self.parser = LogParser()
    
    def get_log_path(self, zone: str, client: str, app: str, version: str, sub_version: str) -> str:
        """Get the path to logs based on parameters"""
        return os.path.join(self.root_dir, zone, client, app, version, sub_version)
    
    def read_logs(self, zone: str, client: str, app: str, version: str, sub_version: str) -> tuple:
        """Read logs from specified location"""
        base_path = self.get_log_path(zone, client, app, version, sub_version)
        
        if not os.path.exists(base_path):
            return None, f"Log path does not exist: {base_path}"
        
        # Find all log files
        log_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(('.log', '.error', '.info', '.debug', '.txt')):
                    log_files.append(os.path.join(root, file))
        
        if not log_files:
            return None, f"No log files found in {base_path}"
        
        # Read and parse all logs
        all_logs = []
        structured_logs = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    all_logs.append(f"=== File: {os.path.basename(log_file)} ===\n{content}")
                    
                    # Parse structured logs
                    for line in content.split('\n'):
                        if line.strip():
                            parsed = self.parser.parse_line(
                                line, zone, client, app, f"{version}/{sub_version}"
                            )
                            if parsed:
                                structured_logs.append(parsed)
            except Exception as e:
                all_logs.append(f"Error reading {log_file}: {str(e)}")
        
        return {
            "raw": "\n".join(all_logs),
            "structured": structured_logs,
            "file_count": len(log_files),
            "zone": zone,
            "client": client,
            "app": app,
            "version": f"{version}/{sub_version}"
        }, None
    
    def get_available_logs(self) -> Dict:
        """Scan and return available log structure"""
        structure = {}
        
        if not os.path.exists(self.root_dir):
            return structure
        
        for zone in os.listdir(self.root_dir):
            zone_path = os.path.join(self.root_dir, zone)
            if os.path.isdir(zone_path):
                structure[zone] = {}
                for client in os.listdir(zone_path):
                    client_path = os.path.join(zone_path, client)
                    if os.path.isdir(client_path):
                        structure[zone][client] = {}
                        for app in os.listdir(client_path):
                            app_path = os.path.join(client_path, app)
                            if os.path.isdir(app_path):
                                structure[zone][client][app] = []
                                for version in os.listdir(app_path):
                                    version_path = os.path.join(app_path, version)
                                    if os.path.isdir(version_path):
                                        structure[zone][client][app].append(version)
        
        return structure