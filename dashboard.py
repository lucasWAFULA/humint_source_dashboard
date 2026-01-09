import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Human Source Dashboard", layout="wide")

st.title("Human Source Tasking and Performance Dashboard")

st.header("Optimized Task Assignments (Stage 1)")
stage1_data = pd.DataFrame(
    np.random.randint(0, 2, (6, 5)),
    columns=[f"Task {i+1}" for i in range(5)],
    index=[f"Source {i+1}" for i in range(6)]
)
st.dataframe(stage1_data)

st.header("Behavior Realisation and Recourse (Stage 2)")
fig, ax = plt.subplots()
stage1_data.sum().plot(kind="bar", ax=ax)
st.pyplot(fig)

st.header("Risk Exposure by Task")
risk = np.random.rand(5)
fig2, ax2 = plt.subplots()
ax2.bar(stage1_data.columns, risk)
st.pyplot(fig2)

st.header("Source Behavior Profiles")
behavior = pd.DataFrame({
    "Cooperative": np.random.rand(6),
    "Deceptive": np.random.rand(6),
    "Neutral": np.random.rand(6)
}, index=stage1_data.index)
st.dataframe(behavior)

st.header("Value Comparison: ML vs TSSP")
comparison = pd.DataFrame({
    "ML": np.random.rand(5),
    "TSSP": np.random.rand(5)
}, index=[f"Metric {i+1}" for i in range(5)])
st.line_chart(comparison)
