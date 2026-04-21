from datetime import datetime, timedelta

def correlate_errors_by_time(
    structured_logs,
    window_minutes=1,
    threshold=2
):
    """
    Phase-3: Time-based correlation
    Detects if `threshold` number of errors occur within time window
    """

    if not structured_logs or len(structured_logs) < threshold:
        return {
            "correlated": False,
            "message": "No significant time-based correlation detected"
        }

    window = timedelta(minutes=window_minutes)

    # Convert timestamps safely
    timestamps = []
    for log in structured_logs:
        try:
            timestamps.append(datetime.fromisoformat(log.timestamp))
        except Exception:
            continue

    timestamps.sort()

    count = 1
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] <= window:
            count += 1
            if count >= threshold:
               return {
                    "correlated": True,
                    "message": f"{threshold} errors occurred within {window_minutes} minute(s)",
                    "reason": "Time-based anomaly detected"
                }

        else:
            count = 1

    return {
        "correlated": False,
        "message": "No significant time-based correlation detected"
    }




