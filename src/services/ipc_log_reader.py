import os

def read_ipc_logs(ipc_root="ipc/log"):
    logs = {}

    if not os.path.exists(ipc_root):
        return logs

    for service in os.listdir(ipc_root):
        service_dir = os.path.join(ipc_root, service)
        if os.path.isdir(service_dir):
            log_file = os.path.join(service_dir, f"{service}.log")
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    logs[service] = f.readlines()

    return logs
