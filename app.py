# dashboard/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Week1 Dashboard", layout="wide")
st.title("Week 1: Cohort / Models / Fairness")

@st.cache_data
def load_interim(path="data/interim/heart_cleveland_interim.csv"):
    return pd.read_csv(path)

# Sidebar
st.sidebar.header("Data / Options")
data_path = st.sidebar.text_input("Interim CSV path", value="data/interim/heart_cleveland_interim.csv")
if st.sidebar.button("Reload Data"):
    load_interim.clear()

# Tabs
tabs = st.tabs(["Cohort", "Models", "Fairness"])

with tabs[0]:
    st.header("Cohort")
    try:
        df = load_interim(data_path)
    except Exception as e:
        st.error(f"Couldn't load interim dataset at {data_path}: {e}")
        st.stop()

    st.write("Dataset shape:", df.shape)
    st.subheader("Descriptive stats")
    target_col = "y" if "y" in df.columns else ("target" if "target" in df.columns else None)

if target_col is None:
    st.error("No target column found. Expected 'y' or 'target'. Please check your dataset.")
else:
    available_cols = [c for c in ["age","sex","trestbps","chol","thalach",target_col] if c in df.columns]
    st.write(df[available_cols].describe())

    prevalence = df[target_col].mean()
    st.metric(label="Outcome prevalence", value=f"{prevalence:.2%}")

  


    st.subheader("Outcome prevalence")
    prevalence = df['y'].mean()
    st.metric("Outcome prevalence", f"{prevalence:.3f}")

    st.subheader("Age distribution")
    fig, ax = plt.subplots()
    ax.hist(df['age'], bins=15)
    st.pyplot(fig)

with tabs[1]:
    st.header("Models")
    st.write("This tab will show baseline model metrics and plots.")
    st.info("Baseline Logistic Regression metrics will appear here once trained.")
    # placeholder: we could load saved model and show AUROC if exists
    import os
    if os.path.exists("reports/baseline_lr.joblib"):
        st.success("Found saved baseline model: reports/baseline_lr.joblib")
    else:
        st.warning("No saved model found (reports/baseline_lr.joblib). Run the baseline notebook first.")

with tabs[2]:
    st.header("Fairness / Subgroups")
    st.write("Subgroup sizes and outcome rates.")
    try:
        subgroup = pd.read_csv("reports/subgroup_table.csv")
        st.dataframe(subgroup)
    except Exception as e:
        st.warning("subgroup table not found. Run baseline notebook to generate reports/subgroup_table.csv.")
