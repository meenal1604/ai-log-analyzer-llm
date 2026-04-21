# 🚀 AI Log Analyzer with LLM-Based Root Cause Analysis

An intelligent log monitoring system that detects anomalies and automatically identifies root causes using Large Language Models.



## 🎯 Problem Statement
In large-scale systems, manual log analysis is time-consuming, error-prone, and inefficient.



## 💡 Solution
This project automates log analysis using AI to:
- Detect anomalies in logs
- Identify root causes
- Suggest actionable fixes


## 🔥 Key Features
- Automated anomaly detection
- LLM-powered Root Cause Analysis (RCA)
- Severity classification (INFO, WARN, ERROR)
- AI-generated fix recommendations
- Interactive dashboard using Streamlit
- Support for real-time and demo logs


## 🧠 Workflow
1. Log ingestion  
2. Anomaly detection  
3. Pattern analysis  
4. Root cause generation using LLM  
5. Suggested fixes


## 📂 Sample Logs
Demo logs available in `ipc/log/demo/`



## ⚙️ Tech Stack
- Python  
- Streamlit  
- AWS Bedrock (LLM)  
- FAISS (vector search)  
- Pandas

## ▶️ How to Run
-git clone https://github.com/meenal1604/ai-log-analyzer-llm

-cd ai-log-analyzer-llm

-pip install -r requirements.txt

-streamlit run app.py

⚠️ Note:
AWS Bedrock credentials are required for LLM-based analysis.
Without credentials, the application will still run but AI features may be limited.
