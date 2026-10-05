"""
🛰️ Land Usage Classifier
Run:  streamlit run app.py
Put your .pt models in the models/ folder — every model there is used.
"""
import streamlit as st

import style

st.set_page_config(page_title="Land Usage Classifier", page_icon="🛰️", layout="wide")
style.apply()

pages = st.navigation([
    st.Page("views/predict.py", title="Predict", icon="🔍", default=True),
    st.Page("views/dashboard.py", title="Dashboard", icon="📊"),
    st.Page("views/model_info.py", title="Models", icon="🤖"),
])
with st.sidebar:
    st.markdown("### 🛰️ Land Usage")
    st.caption("Satellite image classifier")
pages.run()
