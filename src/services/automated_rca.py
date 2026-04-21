def generate_automated_rca(anomaly_result, time_corr_result, total_errors):
    """
    Generates a final automated RCA summary
    """

    rca_lines = []

    # Error volume
    rca_lines.append(f"Total ERROR events detected: {total_errors}.")

    # Anomaly logic
    if anomaly_result.get("anomaly"):
        rca_lines.append(
            f"Anomaly detected: {anomaly_result.get('reason')}."
        )
    else:
        rca_lines.append("No abnormal error spike detected.")

    # Time correlation logic
    if time_corr_result.get("correlated"):
        rca_lines.append(
            "Multiple errors occurred in close time intervals, indicating a cascading failure."
        )
    else:
        rca_lines.append(
            "Errors are spread over time with no strong temporal correlation."
        )

    # Final RCA conclusion
    rca_lines.append(
        "Root cause is likely due to service instability or configuration issues rather than a single isolated failure."
    )

    return " ".join(rca_lines)
