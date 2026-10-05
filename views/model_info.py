"""🤖 Models — every model in the models/ folder."""
from html import escape

import streamlit as st

import config
import shared
import style

style.header("🤖 Models", f"Every .pt / .pth file in the {config.MODELS_FOLDER}/ folder is used by the website.")

models = shared.get_models()
if not models:
    st.info(f"Put your model files (.pt / .pth) in the **{config.MODELS_FOLDER}/** folder, then refresh this page.")
    st.stop()

ready = [m for m in models if m.ok]
st.markdown(
    '<div class="mcards">' + "".join(
        f'<div class="mcard"><div class="mcard-name">{escape(m.file)}</div>'
        f'<div class="mcard-label">{"✅" if m.ok else "❌"} {escape(m.name)}</div>'
        + (f'<div class="row"><span>Architecture</span><span>{escape(m.arch)}</span></div>'
           f'<div class="row"><span>Classes</span><span>{len(m.classes)}</span></div>'
           f'<div class="row"><span>Input size</span><span>{m.img_size}×{m.img_size}</span></div>'
           f'<div class="row"><span>Parameters</span><span>{sum(p.numel() for p in m.model.parameters()):,}</span></div>'
           if m.ok else '<div class="row"><span>Could not be loaded — see below</span></div>')
        + "</div>" for m in models) + "</div>",
    unsafe_allow_html=True,
)
st.caption(f"{len(ready)} of {len(models)} models ready")

for m in models:
    if not m.ok:
        st.error(f"**{m.file}** could not be loaded\n\n```\n{m.error}\n```")
    for note in m.notes:
        st.warning(f"**{m.file}**: {note}")

if ready:
    st.markdown("#### 🏷️ Classes")
    if not config.CLASS_NAMES:
        st.info("The class names are not set yet — write them in `CLASS_NAMES` in **config.py** "
                "(same order as in training).")
    for m in ready:
        st.markdown(f"**{m.name}**: " + ", ".join(m.classes))
