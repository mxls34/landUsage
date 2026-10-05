"""
🛰️ Land Usage Classifier
Run:  streamlit run app.py
Put your .pt models in the models/ folder — every model there is used.
"""
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config
import loader
import style

st.set_page_config(page_title="Land Usage Classifier", page_icon="🛰️", layout="centered",
                   initial_sidebar_state="collapsed")
style.apply()


@st.cache_resource(show_spinner="Loading model…")
def load_model(file_name: str, modified: float) -> loader.LoadedModel:
    return loader.load(loader.BASE_DIR / config.MODELS_FOLDER / file_name)   # loaded once, then reused


def label(name: str) -> str:
    return f"{config.CLASS_ICONS.get(name, '📍')} {name}"


style.header("🛰️ Land Usage Classifier", "Upload a satellite image and let the models classify the land usage.")

files = loader.model_files()
if not files:
    st.info(f"Put your model files (.pt / .pth) in the **{config.MODELS_FOLDER}/** folder, then refresh this page.")
    st.stop()

models = [load_model(p.name, p.stat().st_mtime) for p in files]
ready = [m for m in models if m.ok]
for m in models:
    if not m.ok:
        st.error(f"Could not load **{m.file}**\n\n```\n{m.error}\n```")
    for note in m.notes:
        st.warning(f"**{m.file}**: {note}")
if not ready:
    st.stop()

# ----------------------------------------------------------------- upload + predict
uploaded = st.file_uploader("Upload one satellite image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
if uploaded is None:
    st.session_state.pop("result", None)
    st.stop()
try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("This file is not a readable image.")
    st.stop()

upload_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}-{uploaded.size}"
c_img, c_btn = st.columns([1, 1], gap="medium", vertical_alignment="center")
c_img.image(image, caption=uploaded.name, width="stretch")
if c_btn.button("✨ Predict", width="stretch"):
    with st.spinner("Predicting…"):
        st.session_state.result = (upload_id, [(m, *loader.predict(m, image)) for m in ready])

if st.session_state.get("result", (None,))[0] != upload_id:
    st.stop()
results = st.session_state.result[1]   # [(model, probabilities, ms), ...]

# ----------------------------------------------------------------- final answer
answers = [m.classes[int(p.argmax())] for m, p, _ in results]
votes = Counter(answers)
top_votes = max(votes.values())
# tie -> the class with the highest total confidence
final = max((a for a in votes if votes[a] == top_votes),
            key=lambda a: sum(p.max() for (m, p, _), b in zip(results, answers) if b == a))
agreeing = [float(p.max()) * 100 for (m, p, _), a in zip(results, answers) if a == final]

if len(results) == 1:
    style.answer_card(label(final), ["Prediction", results[0][0].arch],
                      [f"Confidence: {agreeing[0]:.1f}%"])
else:
    style.answer_card(label(final), ["Final answer (majority vote)", f"{top_votes} of {len(results)} models agree"],
                      [f"Average confidence of these models: {np.mean(agreeing):.1f}%"])

st.markdown("#### 🤖 Each model's answer")
cards = []
for (m, p, ms), answer in zip(results, answers):
    order = np.argsort(p)[::-1][:3]
    cards.append({"name": m.name, "arch": m.arch, "label": label(answer), "ms": ms, "agrees": answer == final,
                  "top": [(m.classes[i], float(p[i]) * 100) for i in order]})
style.model_cards(cards)

# ----------------------------------------------------------------- compare (when the models share their classes)
classes = results[0][0].classes
if len(results) > 1 and all(m.classes == classes for m, _, _ in results):
    z = np.array([p for _, p, _ in results]) * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=classes, y=[m.name for m, _, _ in results], zmin=0, zmax=100, xgap=2, ygap=2,
        colorscale=[[0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"], [0.5, "#3987e5"],
                    [0.67, "#256abf"], [0.83, "#184f95"], [1, "#0d366b"]],
        text=[[f"{v:.0f}%" if v >= 1 else "" for v in row] for row in z], texttemplate="%{text}",
        hovertemplate="<b>%{z:.1f}%</b> %{x}<br>%{y}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, ticksuffix="%"),
    ))
    fig.update_layout(height=140 + 40 * len(results), margin=dict(l=8, r=8, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Poppins, system-ui, sans-serif", size=12, color="#52514e"))
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    with st.container(border=True):
        st.markdown("#### 📊 Probability of every class")
        st.caption("One row per model — darker = more sure")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
