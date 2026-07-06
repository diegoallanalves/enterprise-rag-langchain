import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import streamlit as st
from build_vector_db import build_vector_db
from rag_chain import run_rag_query
from data_quality_checks import run_data_quality_checks


st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")


def apply_theme():
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --surface-muted: #f6f8fb;
            --line: #d9e1ec;
            --text-muted: #526071;
            --accent: #0f766e;
            --accent-soft: #d9f3ef;
            --high: #b42318;
            --high-bg: #fee4e2;
            --medium: #b54708;
            --medium-bg: #fef0c7;
            --low: #027a48;
            --low-bg: #dcfae6;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: #101828;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: #14b8a6;
            color: #06201d;
            font-weight: 700;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: #2dd4bf;
            color: #042f2e;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, #f8fafc 0%, #ecfeff 46%, #f7fee7 100%);
            border-radius: 8px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .hero-title {
            color: #111827;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 0.5rem 0;
        }

        .hero-subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            max-width: 860px;
            margin: 0;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .status-pill {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #344054;
            font-size: 0.82rem;
            font-weight: 700;
            padding: 0.35rem 0.7rem;
        }

        .stat-card {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 8px;
            padding: 1rem;
            min-height: 112px;
        }

        .stat-label {
            color: var(--text-muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .stat-value {
            color: #101828;
            font-size: 2rem;
            font-weight: 850;
            line-height: 1.05;
        }

        .stat-note {
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }

        .high-card {
            border-left: 5px solid var(--high);
            background: linear-gradient(180deg, #ffffff 0%, var(--high-bg) 160%);
        }

        .medium-card {
            border-left: 5px solid var(--medium);
            background: linear-gradient(180deg, #ffffff 0%, var(--medium-bg) 160%);
        }

        .low-card {
            border-left: 5px solid var(--low);
            background: linear-gradient(180deg, #ffffff 0%, var(--low-bg) 160%);
        }

        .neutral-card {
            border-left: 5px solid var(--accent);
            background: linear-gradient(180deg, #ffffff 0%, var(--accent-soft) 160%);
        }

        .section-title {
            color: #101828;
            font-size: 1.2rem;
            font-weight: 800;
            margin: 1.1rem 0 0.5rem 0;
        }

        .source-box {
            border: 1px solid var(--line);
            background: var(--surface-muted);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.65rem;
        }

        .source-title {
            color: #101828;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .source-meta {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Enterprise RAG AI Data Quality Assistant</div>
            <p class="hero-subtitle">
                A business-focused review workspace for sales orders, inventory risk,
                customer-name quality issues, retrieved policy context, and AI-generated recommendations.
            </p>
            <div class="status-row">
                <span class="status-pill">LangChain</span>
                <span class="status-pill">ChromaDB</span>
                <span class="status-pill">OpenAI</span>
                <span class="status-pill">Streamlit Dashboard</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label, value, note, class_name):
    st.markdown(
        f"""
        <div class="stat-card {class_name}">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def print_quality_summary(df, answer=None):
    risk_counts = df["AI_Risk_Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    issue_counts = {
        "Stock shortages": int(df["Stock_Shortage"].sum()),
        "Promotion supply risks": int(df["Promotion_Supply_Risk"].sum()),
        "Promotion shortage risks": int(df["Promotion_Shortage_Risk"].sum()),
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
        "Promotion_Supply_Risk",
        "Promotion_Shortage_Risk",
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
    risk_counts = df["AI_Risk_Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    issue_counts = {
        "Stock shortages": int(df["Stock_Shortage"].sum()),
        "Promotion supply risks": int(df["Promotion_Supply_Risk"].sum()),
        "Promotion shortage risks": int(df["Promotion_Shortage_Risk"].sum()),
        "Customer mismatches": int(df["Customer_Name_Mismatch"].sum()),
        "Missing customer names": int(df["Missing_Customer_Name"].sum()),
    }

    st.markdown('<div class="section-title">Risk Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card("High Risk", int(risk_counts["High"]), "Immediate review", "high-card")
    with col2:
        render_stat_card("Medium Risk", int(risk_counts["Medium"]), "Needs validation", "medium-card")
    with col3:
        render_stat_card("Low Risk", int(risk_counts["Low"]), "No issue detected", "low-card")
    with col4:
        render_stat_card("Total Records", len(df), "Rows analysed", "neutral-card")

    st.markdown('<div class="section-title">Analytical View</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-title">Priority Records</div>', unsafe_allow_html=True)
    priority_columns = [
        "PO_Number",
        "Customer_Name",
        "Product_Name",
        "Region",
        "Stock_Shortage",
        "Promotion_Supply_Risk",
        "Promotion_Shortage_Risk",
        "Restock_Date",
        "Customer_Name_Mismatch",
        "Missing_Customer_Name",
        "AI_Risk_Level",
        "AI_Recommendation",
    ]
    priority_df = df[df["AI_Risk_Level"].isin(["High", "Medium"])][priority_columns]
    st.dataframe(priority_df, use_container_width=True, hide_index=True)

    with st.expander("View all analysed records"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_answer(answer):
    st.markdown('<div class="section-title">AI Recommendation Brief</div>', unsafe_allow_html=True)
    st.markdown(answer)


def show_sources(docs):
    st.markdown('<div class="section-title">Retrieved Sources</div>', unsafe_allow_html=True)
    if not docs:
        st.info("No retrieved sources were returned.")
        return

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        po_number = doc.metadata.get("po_number", "policy")
        preview = doc.page_content[:360].strip()
        if len(doc.page_content) > 360:
            preview += "..."
        st.markdown(
            f"""
            <div class="source-box">
                <div class="source-title">Source {index}: {source}</div>
                <div class="source-meta">Reference: {po_number}</div>
                <div class="source-meta">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


apply_theme()
show_header()

with st.sidebar:
    st.header("Workflow")
    st.caption("Prepare the vector database, then run the RAG assistant or the data quality dashboard.")
    if st.button("Build Vector Database", type="primary"):
        with st.spinner("Building vector database..."):
            build_vector_db()
        st.success("Vector DB created.")

    st.divider()
    st.caption("Tip: the data quality dashboard works without calling the LLM.")

question = st.text_input(
    "Ask a business question",
    "Which sales orders have inventory or data quality risks, and what actions should the business take?",
)

action_col1, action_col2 = st.columns([1, 1])
run_rag = action_col1.button("Run RAG Assistant", type="primary", use_container_width=True)
run_checks = action_col2.button("Run Data Quality Checks Only", use_container_width=True)

if run_rag:
    with st.spinner("Running RAG..."):
        answer, docs, df = run_rag_query(question)
    print_quality_summary(df, answer)

    answer_tab, dashboard_tab, sources_tab, data_tab = st.tabs(
        ["AI Brief", "Dashboard", "Retrieved Context", "Full Data"]
    )
    with answer_tab:
        show_answer(answer)
    with dashboard_tab:
        show_quality_dashboard(df)
    with sources_tab:
        show_sources(docs)
    with data_tab:
        st.dataframe(df, use_container_width=True, hide_index=True)

if run_checks:
    df = run_data_quality_checks()
    print_quality_summary(df)
    show_quality_dashboard(df)
