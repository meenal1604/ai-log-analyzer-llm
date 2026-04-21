def generate_rca(correlated_events):
    rca_reports = []

    for time_slot, events in correlated_events.items():
        affected_services = set(e["service"] for e in events)

        rca_reports.append({
            "time_window": str(time_slot),
            "affected_services": list(affected_services),
            "issue_summary": "Multiple services reported anomalies simultaneously",
            "probable_root_cause": "Downstream dependency failure or network latency",
            "recommended_action": [
                "Check network connectivity",
                "Verify dependent services",
                "Restart affected components if required"
            ]
        })

    return rca_reports
