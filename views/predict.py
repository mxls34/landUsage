"""🔍 Predict — upload one picture from your device, every model classifies it."""
from collections import Counter

import numpy as np
import streamlit as st
from PIL import Image

import charts
import config
import loader
import shared
import style

style.header("🔍 Predict", "Upload a satellite image from your device and let the models classify the land usage.")

models = shared.get_models()
ready = [m for m in models if m.ok]
if not models:
    st.info(f"Put your model files (.pt / .pth) in the **{config.MODELS_FOLDER}/** folder, then refresh this page.")
    st.stop()
if not ready:
    st.error("No model could be loaded — open the **Models** page to see why.")
    st.stop()
if len(ready) < len(models):
    st.caption(f"⚠️ {len(models) - len(ready)} model(s) could not be loaded — the **Models** page shows why.")

# ----------------------------------------------------------------- upload + predict
uploaded = st.file_uploader("📤 Upload a picture — press the button and choose one from your device",
                            type=shared.IMAGE_TYPES)
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
shared.save_prediction(upload_id, uploaded.name, image, results, final)   # for the Dashboard page

if len(results) == 1:
    style.answer_card(shared.label(final), ["Prediction", results[0][0].arch], [f"Confidence: {agreeing[0]:.1f}%"])
else:
    style.answer_card(shared.label(final), ["Final answer (majority vote)", f"{top_votes} of {len(results)} models agree"],
                      [f"Average confidence of these models: {np.mean(agreeing):.1f}%"])

st.markdown("#### 🤖 Each model's answer")
cards = []
for (m, p, ms), answer in zip(results, answers):
    order = np.argsort(p)[::-1][:3]
    cards.append({"name": m.name, "arch": m.arch, "label": shared.label(answer), "ms": ms, "agrees": answer == final,
                  "top": [(m.classes[i], float(p[i]) * 100) for i in order]})
style.model_cards(cards)
st.page_link("views/dashboard.py", label="Compare the models on the Dashboard", icon="📊")

# ----------------------------------------------------------------- compare (when the models share their classes)
classes = results[0][0].classes
if len(results) > 1 and all(m.classes == classes for m, _, _ in results):
    z = np.array([p for _, p, _ in results]) * 100
    with st.container(border=True):
        st.markdown("#### 📊 Probability of every class")
        st.caption("One row per model — darker = more sure")
        st.plotly_chart(charts.heatmap(z, classes, [m.name for m, _, _ in results],
                                       text=[[f"{v:.0f}%" if v >= 1 else "" for v in row] for row in z],
                                       hover="<b>%{z:.1f}%</b> %{x}<br>%{y}<extra></extra>",
                                       colorbar_title="Probability"),
                        width="stretch", config=charts.CONFIG)
