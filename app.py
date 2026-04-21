import streamlit as st
import plotly.graph_objects as go  
import plotly.express as px
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import yaml
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import boto3
import json

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)
def analyze_log_with_llm(log_text):
    prompt = f"""
    Analyze the following log and tell:
    1. Is it anomaly or normal?
    2. Root cause
    3. Suggested fix

    Log:
    {log_text}
    """

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 200
    })

    response = bedrock_client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=body
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]
def detect_error_anomaly(structured_logs, threshold=5):
    # Count error logs
    error_lines = [
        log for log in structured_logs
        if "error" in str(log).lower()
    ]

    error_count = len(error_lines)

    if error_count > threshold:
        explanation = f"""
An anomaly has been detected in the system logs.

Reason:
The observed error count ({error_count}) exceeded the predefined threshold ({threshold}).

This indicates abnormal system behavior such as an error spike,
unexpected failures, or instability in system operations.
"""
        anomaly_flag = True
    else:
        explanation = f"""
No anomaly detected.

The total error count ({error_count}) is within the acceptable threshold ({threshold}),
indicating normal system behavior.
"""
        anomaly_flag = False

    return {
        "anomaly_detected": anomaly_flag,
        "message": explanation,
        "error_count": error_count,
        "threshold": threshold
    }
def detect_log_anomalies(logs):
    # Convert logs to numerical vectors
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(logs)

    # Train Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)

    predictions = model.predict(X)

    # -1 = anomaly, 1 = normal
    results = []
    for log, pred in zip(logs, predictions):
        results.append({
            "log": log,
            "anomaly": "YES" if pred == -1 else "NO"
        })

    return results

load_dotenv()

print(os.getenv("AWS_ACCESS_KEY_ID"))

def extract_section(text, section):
    try:
        start = text.index(section) + len(section)
        end = min(
            [text.index(s, start) for s in [
                "Response:",
                "What happened:",
                "Possible root cause:",
                "Recommended fix:",
                "Additional considerations:"
            ] if s != section and s in text] + [len(text)]
        )
        return text[start:end].strip()
    except ValueError:
        return ""



from src.services.log_reader import LogReader
from src.services.rag_engine import RAGEngine
from src.services.knowledge_base import KnowledgeBase
from src.services.template_rca import TemplateRCA
from src.services.anomaly_detector import detect_error_anomaly
from src.services.time_correlation import correlate_errors_by_time
from src.services.automated_rca import generate_automated_rca

# ... after imports ...

def analyze_logs(log_data, query=None, rag_engine=None, kb=None):
    """
    Analyze logs and return structured results
    """
    

    results = {
        'log_stats': {},
        'log_data': log_data,
        'exact_matches': [],
        'similar_errors': [],
        'error_lines': [],
        'kb_solutions': [],
        'solutions': []
    }
   # -------------------------
# PROMOTE KB → RECOMMENDED FIX
# -------------------------
    results['solutions'] = []   # make sure it's initialized

    for sol in results['kb_solutions']:
        results['solutions'].append({
        "error": sol.get("error_type", "Known Issue"),
        "solution": "\n".join(sol.get("solution_steps", [])),
        "confidence": sol.get("confidence", "Medium"),
        "severity": sol.get("severity", "Medium"),
        "prevention": sol.get("prevention", "Follow standard best practices"),
        "exact_match": True
    })
   

    if not log_data or 'structured' not in log_data:
        return results

    # -------------------------
    # BASIC STATS
    # -------------------------
    structured_logs = log_data.get('structured', [])

    total_errors = sum(1 for log in structured_logs if log.log_level == 'ERROR')
    unique_components = set(log.component for log in structured_logs if log.component)

    results['log_stats'] = {
        'file_count': log_data.get('file_count', 0),
        'total_errors': total_errors,
        'unique_components': len(unique_components)
    }

    # Extract error lines
    results['error_lines'] = [
        str(log) for log in structured_logs if log.log_level == 'ERROR'
    ]

    # -------------------------
    # RAG SEARCH
    # -------------------------
    if query and rag_engine:
        raw_log_text = log_data.get('raw', '')

        exact_matches = rag_engine.find_exact_matches(query, raw_log_text)
        results['exact_matches'] = exact_matches

        if not exact_matches:
            similar = rag_engine.find_similar_errors(query, raw_log_text)
            results['similar_errors'] = similar[:5]

    # -------------------------
    # KNOWLEDGE BASE SEARCH
    # -------------------------
    if kb and query:
        kb_solutions = []

        # 1️⃣ PRIMARY: semantic search using user query
        query_fixes = kb.search_similar_issues(query, top_k=3)

        for fix in query_fixes:
            kb_solutions.append({
                "error_type": fix["issue"],
                "root_cause": fix["root_cause"],
                "solution_steps": [
                    step.strip()
                    for step in fix["solution"].split(".")
                    if step.strip()
                ],
                "confidence": fix["confidence"]
            })

        # 2️⃣ SECONDARY: component-based search
        for component in list(unique_components)[:2]:
            component_fixes = kb.search_by_component(component)
            if component_fixes:
                kb_solutions.extend(component_fixes)

        # Save KB results
        results['kb_solutions'] = kb_solutions[:5]

        # -------------------------
        # PROMOTE KB → RECOMMENDED FIX
        # -------------------------
        for sol in results['kb_solutions']:
            results['solutions'].append({
                "error": sol.get("error_type", "Known Issue"),
                "solution": "\n".join(sol.get("solution_steps", [])),
                "confidence": sol.get("confidence", "Medium"),
                "exact_match": True
            })

    return results



# Page config
st.set_page_config(
    page_title="LogSentry AI - Enterprise RCA Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize services
@st.cache_resource
def init_services():
    return LogReader(), RAGEngine(), KnowledgeBase(),TemplateRCA()

log_reader, rag_engine, kb, template_rca = init_services()

# DEBUG: Check KB load
st.sidebar.write("KB Entries Loaded:", len(kb.entries))


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #374151;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .log-line {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .log-error { background-color: #FEE2E2; color: #DC2626; }
    .log-warn { background-color: #FEF3C7; color: #D97706; }
    .log-info { background-color: #DBEAFE; color: #1D4ED8; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #374151;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">AI-Powered Log Analysis & RCA Assistant</h1>', unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.header(" Analysis Parameters")

with st.sidebar:
    st.markdown("### 🤖 LLM Status")

    if st.session_state.get("llm_ready", False):
        st.success("LLM Connected")
    else:
        st.warning("LLM Not Connected")

    
    # Get available log structure
    log_structure = log_reader.get_available_logs()
    
    # Zone selection
    zones = list(log_structure.keys()) if log_structure else ["EMEA", "ASIA", "AMERICA"]
    zone = st.selectbox("Zone", zones, index=0 if zones else 0)
    
    # Client selection
    clients = list(log_structure.get(zone, {}).keys()) if zone in log_structure else ["Barclays", "HSBC", "JPMorgan"]
    client = st.selectbox(" Client", clients, index=0 if clients else 0)
    
    # Application selection
    apps = list(log_structure.get(zone, {}).get(client, {}).keys()) if zone in log_structure and client in log_structure[zone] else ["Unigy", "Pulse", "Touch"]
    app = st.selectbox(" Application", apps, index=0 if apps else 0)
    
    # Version selection
    versions = log_structure.get(zone, {}).get(client, {}).get(app, []) if zone in log_structure and client in log_structure[zone] and app in log_structure[zone][client] else ["4.0", "3.0"]
    version = st.selectbox(" Version", versions, index=0 if versions else 0)
    
    # Sub-version selection
    sub_versions = ["4.0.1", "3.0.1", "4.0.0", "3.0.0"]  # This could be dynamic
    sub_version = st.selectbox(" Sub-Version", sub_versions)
    
    # Time range
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input("Start Time", "00:00")
    with col2:
        end_time = st.text_input("End Time", "23:59")
    
    # Query input
    st.markdown("---")
    query = st.text_area(
        " Enter your query or error description:",
        "Why did the system fail with timeout errors?",
        height=100
    )
    
    # Advanced options
    with st.expander(" Advanced Options"):
        top_k = st.slider("Top K results", 1, 10, 5)
        min_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.6)
        include_kb = st.checkbox("Include KB fixes", True)
        index_logs = st.checkbox("Index logs for future", True)
    
    # Action button
    analyze_btn = st.button(" Analyze Logs", type="primary", use_container_width=True)
    
    # Quick queries
    st.markdown("---")
    st.markdown("** Quick Queries:**")
    quick_queries = [
        "Find all timeout errors",
        "Show database connection issues",
        "Analyze 500 Internal Server Errors",
        "Check configuration mismatches"
    ]
    for q in quick_queries:
        if st.button(q, use_container_width=True):
            query = q
            st.rerun()

# Main content
if analyze_btn or 'results' in st.session_state:
    if analyze_btn:
        with st.spinner(" Reading logs..."):
            log_data, error = log_reader.read_logs(zone, client, app, version, sub_version)
            
            if error:
                st.error(f"Error: {error}")
                st.stop()
            
            with st.spinner(" Analyzing logs..."):
                # First, get basic analysis results
                results = analyze_logs(
                    log_data=log_data,
                    query=query,
                    rag_engine=rag_engine,
                    kb=kb
                )
                results["template_rca"] = template_rca.generate(
                    log_data.get("structured", [])
                )
                # 🔎 Phase 3 – Anomaly Detection
                anomaly_result = detect_error_anomaly(
                    structured_logs=log_data.get("structured", []),
                    threshold=5
                )

                results["anomaly"] = anomaly_result
                #  LLM Explanation (Bedrock Claude)
                if anomaly_result.get("anomaly_detected"):
                    try:
                        sample_log = "\n".join(results.get("error_lines", [])[:5]) or "ERROR: Unknown issue"
                        llm_output = analyze_log_with_llm(sample_log)
                        results["llm_explanation"] = llm_output
                        st.session_state["llm_ready"] = True
                    except Exception as e:
                        results["llm_explanation"] = f"LLM Error: {str(e)}"
                        st.session_state["llm_ready"] = False
                else:
                    results["llm_explanation"] = "No anomaly detected, so no AI explanation generated."
                
                # ⏱ Phase 3 – Time-Based Correlation
                time_corr = correlate_errors_by_time(
                    structured_logs=log_data.get("structured", []),
                    window_minutes=5,
                    threshold=3
                )

                results["time_correlation"] = time_corr

                # 🧠 Phase 3 – Automated RCA
                auto_rca = generate_automated_rca(
                    anomaly_result=results.get("anomaly", {}),
                    time_corr_result=results.get("time_correlation", {}),
                    total_errors=results.get("log_stats", {}).get("total_errors", 0)
                )

                results["automated_rca"] = auto_rca


                # Then, get RCA from RAG engine

                rca_result = rag_engine.process_query(
                    query=query,
                    log_data=log_data,
                    zone=zone,
                    client=client
                )

            

                
                # Merge all results
                results.update({
                    'rca': rca_result.get('rca', 'No RCA generated'),
                    'log_stats': {
                         **results.get('log_stats', {}),
                         'query_matches': rca_result.get('log_stats', {}).get('query_matches', 0)
                     }
                })

                # ✅ DO NOT overwrite KB fixes
                if not results.get('solutions'):
                  results['solutions'] = rca_result.get('solutions', [])

                
                st.session_state.results = results
                st.session_state.log_data = log_data
                
                st.success(f" Analysis complete! Found {len(log_data['structured'])} log entries")
    else:
        results = st.session_state.results
        log_data = st.session_state.log_data
    
    # Display results in tabs
   # --- Tabs definition ---

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "RCA Summary",
        "Analytics",
        "Evidence",
        "KB Fixes",
        "Log Details",
        "AI Explanation",
    ])

# --- Tab usage ---

    with tab6:
        st.subheader("🤖 AI Explanation")

        if results.get("llm_explanation"):
            st.success("LLM Reasoning Active")
            st.markdown(results["llm_explanation"])
        else:
            st.info("Run analysis to generate AI explanation.")
   
    
    with tab1:
        st.subheader("🧠 Automated Root Cause Analysis")

        if results.get("rca"):
            st.success("Root Cause Identified")
            st.markdown("### Root Cause")
            st.write(results["rca"])

            impact = results["rca"].get("impact", "N/A") if isinstance(results["rca"], dict) else "N/A"
            affected = (
                ", ".join(results["rca"].get("affected_services", []))
                if isinstance(results["rca"], dict)
                else "N/A"
            )

            st.markdown(f"**Impact:** {impact}")
            st.markdown(f"**Affected Services:** {affected}")

        else:
            st.warning("No RCA generated")

    # RCA Summary
        
        # RCA Summary
        st.markdown("##  **Troubleshooting Results**")
        # ⏱ Time-Based Correlation Result
        if "time_correlation" in results:
            st.markdown("### ⏱ Time-Based Correlation")

            if results["time_correlation"]["correlated"]:
                st.warning(results["time_correlation"]["message"])
            else:
                st.success(results["time_correlation"]["message"])

        # 🤖 Automated RCA Summary
        if "automated_rca" in results:
                st.markdown("## 🤖 Automated RCA Summary")
                st.info(results["automated_rca"])

        # Query section
        st.markdown("###  What You Asked")
        st.info(f"**Query:** \"{query}\"")
        
        # What We Found section
        st.markdown("###  What We Found")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Error Lines", results['log_stats']['total_errors'])
        with col2:
            st.metric("Log Files", results['log_stats']['file_count'])
        with col3:
            st.metric("Unique Error Types", len(set(results.get('error_lines', []))))
        
    # 🚨 Anomaly Detection Result
    if "anomaly" in results:
        st.markdown("### 🚨 Anomaly Detection")

    if results["anomaly"]["anomaly_detected"]:
        st.error(results["anomaly"]["message"])
    else:
        st.success(results["anomaly"]["message"])

        st.write(
            f"Error Count: {results['anomaly']['error_count']} | "
        f"Threshold: {results['anomaly']['threshold']}"
        )

        # Errors Found section - SCROLLABLE GREEN TEXT
        st.markdown("###  Errors Found")
        
        # Get error lines (prioritize exact matches, then similar errors)
        error_lines = []
        if 'exact_matches' in results and results['exact_matches']:
            error_lines = results['exact_matches']
            st.success(f" Found {len(error_lines)} exact matches")
        elif 'similar_errors' in results and results['similar_errors']:
            error_lines = results['similar_errors']
            st.warning(f" Found {len(error_lines)} similar errors")
        
        if error_lines:
            # Create a scrollable container for error lines
            max_height = 300  # Maximum height in pixels
            error_text = ""
            
            for i, line in enumerate(error_lines[:10], 1):  # Show first 10 errors
                # Truncate long lines for the display
                display_line = line
                if len(line) > 100:
                    display_line = line[:100] + "..."
                error_text += f"{i}. {display_line}\n"
            
            # Create scrollable text area
            st.text_area(
                "Error Details",
                value=error_text,
                height=min(max_height, 30 + len(error_lines) * 25),  # Dynamic height
                key="error_display",
                disabled=True,  # Read-only
                label_visibility="collapsed"  # Hide the label
            )
            
            # Show full error details in expandable sections
            st.markdown("** Full Error Details:**")
            for i, line in enumerate(error_lines[:5], 1):  # Show first 5 full errors
                with st.expander(f"Error #{i}", expanded=False):
                    st.code(line, language='text')
        else:
            st.info("No matching errors found")
        
        # Recommended Fix section
        st.markdown("###  Recommended Fix")
        
        if 'solutions' in results and results['solutions']:
            for i, sol in enumerate(results['solutions'][:2], 1):  # Show first 2 solutions
                with st.container():
                    st.markdown(f"**{sol.get('error', 'Issue')}**")
                    if sol.get('exact_match', False):
                        st.success(" **Exact match from Knowledge Base**")
                    
                    solution_text = sol.get('solution', '')
                    if solution_text:
                        # Format as numbered list
                        lines = solution_text.split('\n')
                        for j, step in enumerate(lines, 1):
                            if step.strip():
                                st.write(f"{j}. {step.strip()}")
                    st.markdown("---")
        else:
            st.info("No specific solution found in Knowledge Base")
            st.markdown("### 📄 Template RCA (Phase-3)")
            st.code(results.get("template_rca", "No RCA generated"))


    with tab2:
        st.subheader("📊 Log Analytics")

        if results.get("structured_logs"):
            df = pd.DataFrame(results["structured_logs"])

            if "severity" in df.columns:
                severity_count = df["severity"].value_counts().reset_index()
                severity_count.columns = ["Severity", "Count"]

                fig = px.bar(
                    severity_count,
                    x="Severity",
                    y="Count",
                    title="Log Severity Distribution",
                    color="Severity"
                )
                st.plotly_chart(fig, use_container_width=True)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                time_series = df.groupby(df["timestamp"].dt.hour).size().reset_index(name="count")

                fig2 = px.line(
                    time_series,
                    x="timestamp",
                    y="count",
                    title="Logs Over Time (Hourly)"
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("No analytics available")

    # Analytics Dashboard
        st.subheader(" **Log Analysis Dashboard**")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Create metrics row - HANDLE BOTH OLD AND NEW FORMATS
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Log files
                if 'log_data' in st.session_state:
                    st.metric(" Files", st.session_state.log_data.get('file_count', 0))
                else:
                    st.metric(" Files", 0)
            
            with col2:
                # Total errors - handle both formats
                if 'log_stats' in results and 'total_errors' in results['log_stats']:
                    # Old format
                    st.metric(" Total Errors", results['log_stats']['total_errors'])
                elif 'error_lines' in results:
                    # New simplified format
                    st.metric(" Total Errors", len(results.get('error_lines', [])))
                else:
                    st.metric(" Total Errors", 0)
            
            with col3:
                # Components affected
                if 'log_stats' in results and 'unique_components' in results['log_stats']:
                    st.metric(" Components", results['log_stats']['unique_components'])
                else:
                    # Count from error lines if available
                    if 'exact_matches' in results and results['exact_matches']:
                        import re
                        components = set()
                        for line in results['exact_matches']:
                            match = re.search(r'Component=([A-Za-z]+)', line)
                            if match:
                                components.add(match.group(1))
                        st.metric(" Components", len(components))
                    else:
                        st.metric(" Components", 0)
            
            with col4:
                # Confidence
                if 'exact_matches' in results and results['exact_matches']:
                    st.metric(" Confidence", "High")
                elif 'similar_errors' in results and results['similar_errors']:
                    st.metric(" Confidence", "Medium")
                else:
                    st.metric(" Confidence", "Low")
            
            # Visualization section
            st.subheader(" **Error Distribution**")
            
            if 'log_data' in st.session_state and 'structured' in st.session_state.log_data:
                df = pd.DataFrame([entry.to_dict() for entry in st.session_state.log_data['structured']])
                
                if not df.empty:
                    # Create two columns for charts - RESTORED PIE CHART
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Error types pie chart - RESTORED FROM OLD VERSION
                        if 'error_code' in df.columns and df['error_code'].notna().any():
                            error_counts = df['error_code'].value_counts().head(8)
                            if not error_counts.empty:
                                fig1 = go.Figure(data=[go.Pie(
                                    labels=error_counts.index,
                                    values=error_counts.values,
                                    hole=0.3,
                                    marker_colors=px.colors.sequential.RdBu
                                )])
                                fig1.update_layout(
                                    title="Top Error Types",
                                    showlegend=True,
                                    height=400
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            else:
                                st.info("No error codes found in logs")
                        else:
                            st.info("Error code data not available")
                    
                    with chart_col2:
                        # Component bar chart
                        if 'component' in df.columns:
                            comp_counts = df['component'].value_counts().head(10)
                            if not comp_counts.empty:
                                fig2 = px.bar(
                                    x=comp_counts.index,
                                    y=comp_counts.values,
                                    title="Most Affected Components",
                                    labels={'x': 'Component', 'y': 'Error Count'},
                                    color=comp_counts.values,
                                    color_continuous_scale='Viridis'
                                )
                                fig2.update_layout(height=400)
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("No component data available")
                        else:
                            st.info("Component data not available")
                    
                    # ADD TIME SERIES SECTION BACK
                    st.subheader(" Error Timeline")
                    if 'timestamp' in df.columns and df['timestamp'].notna().any():
                        try:
                            # Convert timestamps
                            df['time_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
                            df = df.dropna(subset=['time_parsed'])
                            
                            if not df.empty:
                                # Group by hour
                                df['hour'] = df['time_parsed'].dt.floor('H')
                                hourly_counts = df.groupby('hour').size().reset_index(name='count')
                                
                                fig3 = px.line(
                                    hourly_counts,
                                    x='hour',
                                    y='count',
                                    title="Errors Over Time (Last 24 Hours)",
                                    markers=True
                                )
                                fig3.update_layout(
                                    xaxis_title="Time",
                                    yaxis_title="Error Count",
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig3, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create timeline: {str(e)}")
                    
                    # Error Severity Breakdown
                    st.subheader(" Error Severity")
                    if 'log_level' in df.columns:
                        severity_counts = df['log_level'].value_counts()
                        if not severity_counts.empty:
                            col1, col2, col3 = st.columns(3)
                            levels = {'ERROR': '🔴', 'WARN': '🟡', 'INFO': '🟢'}
                            
                            for level, emoji in levels.items():
                                count = severity_counts.get(level, 0)
                                if level == 'ERROR':
                                    with col1:
                                        st.metric(f"{emoji} Critical Errors", count)
                                elif level == 'WARN':
                                    with col2:
                                        st.metric(f"{emoji} Warnings", count)
                                elif level == 'INFO':
                                    with col3:
                                        st.metric(f"{emoji} Info Messages", count)
                else:
                    st.info(" No structured log data available for visualization")
            else:
                st.info(" Load logs first using the Analyze button")
        else:
            st.info(" Run an analysis first to see dashboard data")

        
        
    with tab3:
       
    # Evidence
        st.subheader(" **What We Found in Logs**")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            if 'exact_matches' in results:
                exact_matches = results['exact_matches']
                similar_errors = results.get('similar_errors', [])
                
                if exact_matches:
                    st.success(f" Found {len(exact_matches)} EXACT matches for your query")
                    st.subheader(" **Exact Matches**")
                    for i, line in enumerate(exact_matches[:3], 1):
                        with st.expander(f"Match #{i}", expanded=(i==1)):
                            st.code(line)
                elif similar_errors:
                    st.warning(f" No exact matches. Found {len(similar_errors)} similar errors")
                    st.subheader(" **Similar Errors Found**")
                    for i, line in enumerate(similar_errors[:3], 1):
                        with st.expander(f"Similar error #{i}", expanded=(i==1)):
                            st.code(line)
                else:
                    st.info(" No matching errors found")
            
            # Solutions
            if 'solutions' in results and results['solutions']:
                st.subheader(" **Recommended Solution**")
                for sol in results['solutions'][:2]:  # Show max 2 solutions
                    with st.container():
                        st.markdown(f"### **{sol.get('error', 'Issue')}**")
                        if sol.get('exact_match', False):
                            st.success(" **Exact match from Knowledge Base**")
                        
                        solution_text = sol.get('solution', '')
                        if solution_text:
                            steps = solution_text.split('\n')
                            for step in steps:
                                if step.strip():
                                    st.write(f"• {step.strip()}")
                        
                        st.markdown("---")



    with tab4:
        # KB Fixed Log Details
        st.subheader(" **Knowledge Base Solutions**")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Check if we have solutions in results
            if 'kb_solutions' in results and results['kb_solutions']:
                st.success(f" Found {len(results['kb_solutions'])} solutions in Knowledge Base")
                
                # Display each solution
                for i, solution in enumerate(results['kb_solutions'], 1):
                    with st.expander(f"Solution #{i}: {solution.get('error_type', 'Unknown Error')}"):
                        # Solution details
                        st.write(f"**Error Type:** {solution.get('error_type', 'N/A')}")
                        st.write(f"**Component:** {solution.get('component', 'N/A')}")
                        st.write(f"**Confidence:** {solution.get('confidence', 'N/A')}")
                        
                        # Root cause
                        st.write("**Root Cause:**")
                        st.write(solution.get('root_cause', 'No root cause analysis available'))
                        
                        # Solution steps
                        st.write("**Solution Steps:**")
                        solution_steps = solution.get('solution_steps', [])
                        if solution_steps:
                            for j, step in enumerate(solution_steps, 1):
                                st.write(f"{j}. {step}")
                        else:
                            st.write("No specific steps provided")
                        
                        # Prevention
                        if solution.get('prevention'):
                            st.write("**Prevention Tips:**")
                            st.write(solution.get('prevention'))
                        
                        # Related resources
                        if solution.get('resources'):
                            st.write("**Related Resources:**")
                            for resource in solution.get('resources', []):
                                st.write(f"- {resource}")
            
            elif 'exact_matches' in results and results['exact_matches']:
                # If we have exact matches but no KB solutions, search for them
                st.info(" Searching for solutions in Knowledge Base...")
                
                # Placeholder for KB search logic
                # In a real implementation, you would:
                # 1. Extract error patterns
                # 2. Query your knowledge base
                # 3. Display matching solutions
                
                # Example mock data
                with st.expander("Potential Solution: Connection Timeout Error"):
                    st.write("**Error Type:** Database Connection Timeout")
                    st.write("**Root Cause:** Connection pool exhaustion")
                    st.write("**Solution:** Increase connection pool size and add connection validation")
                    st.write("1. Check current connection pool settings")
                    st.write("2. Increase max pool size to 50")
                    st.write("3. Add validation query to connection pool")
                    st.write("4. Monitor connection usage metrics")
            else:
                st.info(" No errors found to search for solutions. Run an analysis first.")
        else:
            st.info(" Run an analysis first to see solutions")



    with tab5:
        # Raw Log Details
        st.subheader(" Raw Log Contents")
        
        # Log level filter
        log_levels = ["ALL", "ERROR", "WARN", "INFO", "DEBUG"]
        selected_level = st.selectbox("Filter by log level", log_levels)
        
        # Display logs with syntax highlighting
        raw_logs = log_data['raw']
        lines = raw_logs.split('\n')
        
        # Create a scrollable log viewer
        log_container = st.container()
        with log_container:
            for line in lines[:config['ui']['max_log_display']]:
                if selected_level == "ALL" or selected_level in line:
                    line_lower = line.lower()
                    if 'error' in line_lower:
                        st.markdown(f'<div class="log-line log-error">{line}</div>', unsafe_allow_html=True)
                    elif 'warn' in line_lower:
                        st.markdown(f'<div class="log-line log-warn">{line}</div>', unsafe_allow_html=True)
                    elif 'info' in line_lower:
                        st.markdown(f'<div class="log-line log-info">{line}</div>', unsafe_allow_html=True)
                    else:
                        st.code(line, language='bash')
        
        if len(lines) > config['ui']['max_log_display']:
            st.warning(f"Showing first {config['ui']['max_log_display']} lines. Total lines: {len(lines)}")

else:
    # Landing page
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3> Fast Analysis</h3>
            <p>Get RCA in seconds using semantic search</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3> AI-Powered</h3>
            <p>RAG architecture with local embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3> Enterprise Ready</h3>
            <p>Multi-zone, multi-client support</p>
        </div>
        """, unsafe_allow_html=True)



   
    # Quick start guide
    st.markdown("---")
    st.markdown("###  Quick Start Guide")
    
    guide_col1, guide_col2 = st.columns(2)
    
    with guide_col1:
        st.markdown("""
        1. **Select parameters** from sidebar
        2. **Enter your query** about the issue
        3. **Click 'Analyze Logs'** button
        4. **Review RCA** in the summary tab
        5. **Check evidence** and KB fixes
        
        **Supported Query Types:**
        - Why did [component] fail?
        - Find all [error type] errors
        - What caused the timeout?
        - Analyze configuration issues
        """)
    
    with guide_col2:
        st.markdown("""
        ** Expected Log Structure:**
        ```
        E:/LogSpace/
        ├── ZONE/
        │   ├── CLIENT/
        │   │   ├── APP/
        │   │   │   ├── VERSION/
        │   │   │   │   ├── SUB_VERSION/
        │   │   │   │   │   ├── *.error
        │   │   │   │   │   └── *.info
        ```
        
        **🔧 Features:**
        - Semantic log search
        - Automatic RCA generation
        - Knowledge base integration
        - Visual analytics
        - Real-time indexing
        """)
    
    # System stats
    st.markdown("---")
    st.markdown("###  System Statistics")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        if hasattr(rag_engine, "vector_store"):
            st.metric("Vector Store Size", f"{rag_engine.vector_store.size()} chunks")
        else:
            st.metric("Vector Store Size", "FAISS / Embeddings Active")
    with stat_col2:
        st.metric("KB Entries", f"{len(kb.entries)} fixes")
    with stat_col3:
        st.metric("Embedding Model", config['embedding']['model_name'])


# Footer
st.markdown("---")
st.markdown("*LogSentry AI v2.0 | RAG-powered Troubleshooting Assistant | Phase 2 Implementation*")