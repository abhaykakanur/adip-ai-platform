import streamlit as st
import pandas as pd
import os


# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="ADIP Dashboard",
    layout="wide"
)


# ---------------- Title ---------------- #

st.title("🚀 Autonomous Data Intelligence Platform")
st.markdown("Real-time AI-powered Business Analytics Dashboard")

st.divider()


# ---------------- File Paths ---------------- #

FEATURE_FILE = "data/features/ecommerce_data.csv"
INSIGHT_FILE = "data/insights/ecommerce_data_insight.txt"
MONITOR_FILE = "monitoring/monitor_report.txt"


# ---------------- Load Data ---------------- #

@st.cache_data
def load_data():
    return pd.read_csv(FEATURE_FILE)


# ---------------- KPIs ---------------- #

st.header("📊 Business KPIs")

if os.path.exists(FEATURE_FILE):

    df = load_data()

    col1, col2, col3 = st.columns(3)

    total_revenue = df["Revenue"].sum()
    avg_revenue = df["Revenue"].mean()
    total_orders = len(df)

    col1.metric("Total Revenue", f"{total_revenue:,.0f}")
    col2.metric("Average Revenue", f"{avg_revenue:.2f}")
    col3.metric("Total Orders", total_orders)

else:
    st.warning("Feature data not found. Run pipeline first.")


st.divider()


# ---------------- Revenue Trend ---------------- #

st.header("📈 Monthly Revenue Trend")

if os.path.exists(FEATURE_FILE):

    if "month" in df.columns:

        trend = df.groupby("month")["Revenue"].sum()

        st.line_chart(trend)

    else:
        st.info("No month column found in data.")

else:
    st.warning("No data available.")


st.divider()


# ---------------- AI Insights ---------------- #

st.header("🧠 AI-Generated Business Insights")

if os.path.exists(INSIGHT_FILE):

    with open(INSIGHT_FILE, "r", encoding="utf-8") as f:

        insights = f.read()

        st.markdown(insights)

else:
    st.info("No AI insights found. Run pipeline with LLM enabled.")


st.divider()


# ---------------- System Health ---------------- #

st.header("📡 System Health & Monitoring")

if os.path.exists(MONITOR_FILE):

    with open(MONITOR_FILE, "r", encoding="utf-8") as f:

        report = f.read()

        st.code(report)

else:
    st.info("No monitoring report found.")


# ---------------- Footer ---------------- #

st.divider()

st.markdown(
    "Built by Abhay Kakanur | Autonomous AI Platform",
    unsafe_allow_html=True
)
