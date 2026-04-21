import os
from datetime import datetime, timedelta
import random

BASE_PATH = "ipc/log"

services = [
    "backup_restore","BCPGrp","bluewave","BRGrp","ds_interzone_adapter",
    "ds_interzone_sync","ha","HAGrp","Kafka","lib_ahcjobexecutor",
    "lib_backup_restore","NotifierGrp","opensips","PresenceGrp",
    "RecordingGrp","service_backup_restore_jobagent","service_license_admin",
    "service_monitoring","service_security_account","service_security_cli",
    "service_security_key","service_uda","service_xmenu","SipAdapterGrp",
    "ua_alert_xlator","UDAGrp","unigy_agent_repository","va","va_app_ds",
    "va_call_history","va_callprocessing","VARegistrarGrp","VATrunksGrp",
    "wamp","zookeeper"
]

levels = ["INFO", "WARN", "ERROR"]

messages = {
    "INFO": "Operation completed successfully",
    "WARN": "Latency threshold exceeded",
    "ERROR": "Service timeout or failure detected"
}

os.makedirs(BASE_PATH, exist_ok=True)

for svc in services:
    svc_path = os.path.join(BASE_PATH, svc)
    os.makedirs(svc_path, exist_ok=True)

    log_file = os.path.join(svc_path, f"{svc}.log")

    start_time = datetime.now() - timedelta(minutes=30)

    with open(log_file, "w") as f:
        for i in range(20):
            ts = start_time + timedelta(minutes=i)
            level = random.choice(levels)
            msg = messages[level]
            f.write(f"{ts} {level} {svc} {msg}\n")

print("✅ IPC log folders and sample logs created successfully")
