Autonomous Data Intelligence Platform (ADIP)

ADIP is an end to end autonomous AI system that converts raw CSV data into meaningful business insights. It automates data engineering, analytics, machine learning, monitoring, and AI powered reporting using intelligent agent orchestration.

The platform is designed to run with minimal human intervention. Once a dataset is provided, the system automatically cleans the data, builds features, trains models, monitors performance, and generates business recommendations.

📌 Features

Automatic data ingestion and profiling

Data cleaning with duplicate removal and missing value handling

Feature engineering and preprocessing

Machine learning model training and evaluation

Best model selection and persistence

Model performance and drift monitoring

AI generated business insights using LLM

Intelligent agent orchestration using LangGraph

Interactive web dashboard using Streamlit

Cloud ready deployment

🏗️ System Architecture
Raw Data → Ingestion → Cleaning → ETL → ML → Monitoring → LLM → Dashboard
                     ↑______________________________________________↓
                               LangGraph Orchestration


The system is controlled by an intelligent routing mechanism that manages execution flow, retries, and fault tolerance.

🛠️ Technology Stack

Programming Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit Learn

Agent Orchestration: LangGraph

Large Language Model: Groq API with Llama models

Web Dashboard: Streamlit

Model Storage: Joblib

Environment Management: Python Dotenv, Virtual Environment

Version Control: Git, GitHub

Cloud Deployment: Render

📁 Project Structure
adip/
│
├── agents/
│   ├── ingestion_agent.py
│   ├── quality_agent.py
│   ├── etl_agent.py
│   ├── analytics_agent.py
│   ├── ml_agent.py
│   ├── monitoring_agent.py
│   └── llm_agent.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── clean/
│   ├── features/
│   ├── reports/
│   └── insights/
│
├── models/
├── monitoring/
├── dashboard.py
├── orchestrator.py
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md

