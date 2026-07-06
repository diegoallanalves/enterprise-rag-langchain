import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import streamlit as st
from build_vector_db import build_vector_db
from rag_chain import run_rag_query
from data_quality_checks import run_data_quality_checks


def print_quality_summary(df, answer=None):
    risk_counts = df["AI_Risk_Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    issue_counts = {
        "Stock shortages": int(df["Stock_Shortage"].sum()),
        "Customer mismatches": int(df["Customer_Name_Mismatch"].sum()),
        "Missing customer names": int(df["Missing_Customer_Name"].sum()),
    }
    risky_records = df[df["AI_Risk_Level"].isin(["High", "Medium"])]
    visible_columns = [
        "PO_Number",
        "Customer_Name",
        "Product_Name",
        "Region",
        "Stock_Shortage",
        "Customer_Name_Mismatch",
        "Missing_Customer_Name",
        "AI_Risk_Level",
        "AI_Recommendation",
    ]

    print("\n" + "=" * 80)
    print("ENTERPRISE RAG AI DATA QUALITY RESULTS")
    print("=" * 80)
    if answer:
        print("\nAI ANSWER")
        print(answer)
    print("\nRISK LEVEL COUNTS")
    print(risk_counts.to_string())
    print("\nISSUES DETECTED")
    for issue, count in issue_counts.items():
        print(f"{issue}: {count}")
    print("\nHIGH AND MEDIUM RISK RECORDS")
    print(risky_records[visible_columns].to_string(index=False))
    print("=" * 80 + "\n")


def show_quality_dashboard(df):
    st.subheader("Data Quality Dashboard")

    risk_counts = df["AI_Risk_Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    issue_counts = {
        "Stock shortages": int(df["Stock_Shortage"].sum()),
        "Customer mismatches": int(df["Customer_Name_Mismatch"].sum()),
        "Missing customer names": int(df["Missing_Customer_Name"].sum()),
    }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("High Risk", int(risk_counts["High"]))
    col2.metric("Medium Risk", int(risk_counts["Medium"]))
    col3.metric("Low Risk", int(risk_counts["Low"]))
    col4.metric("Total Records", len(df))

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.caption("Risk level distribution")
        st.bar_chart(risk_counts)
    with chart_col2:
        st.caption("Issues detected")
        st.bar_chart(issue_counts)

    if "Region" in df.columns:
        st.caption("Risk by region")
        risk_by_region = df.pivot_table(
            index="Region",
            columns="AI_Risk_Level",
            values="PO_Number",
            aggfunc="count",
            fill_value=0,
        )
        st.bar_chart(risk_by_region)

    st.subheader("Data Quality Results")
    st.dataframe(df, use_container_width=True)


st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
st.title("Enterprise RAG AI Data Quality Assistant")

with st.sidebar:
    st.header("Actions")
    if st.button("Build Vector Database"):
        with st.spinner("Building vector database..."):
            build_vector_db()
        st.success("Vector DB created.")

question = st.text_input("Ask a question", "Which sales orders have inventory or data quality risks?")

if st.button("Run RAG Assistant"):
    with st.spinner("Running RAG..."):
        answer, docs, df = run_rag_query(question)
    print_quality_summary(df, answer)
    st.subheader("AI Answer")
    st.write(answer)
    st.subheader("Retrieved Sources")
    for doc in docs:
        st.json(doc.metadata)
    show_quality_dashboard(df)

if st.button("Run Data Quality Checks Only"):
    df = run_data_quality_checks()
    print_quality_summary(df)
    show_quality_dashboard(df)
