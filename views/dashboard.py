"""📊 Dashboard — test every model on the same labelled images and compare them."""
import os

import numpy as np
import pandas as pd
import streamlit as st

import charts
import config
import shared
import style

style.header("📊 Model Comparison Dashboard",
             "Every model is tested on the same labelled images, then compared side by side.")

ready = [m for m in shared.get_models() if m.ok]
if not ready:
    st.info(f"Put your model files (.pt / .pth) in the **{config.MODELS_FOLDER}/** folder — "
            "the **Models** page shows if one could not be loaded.")
    st.stop()

classes = ready[0].classes
models = [m for m in ready if m.classes == classes]
if len(models) < len(ready):
    st.caption("⚠️ Left out (different classes): " + ", ".join(m.name for m in ready if m.classes != classes))

# ----------------------------------------------------------------- test images + ranking
c_source, c_rank = st.columns([2, 1])
source = c_source.radio("Test images", ["📁 Test folder", "📦 Upload a .zip"], horizontal=True)
rank_by = c_rank.selectbox("Rank models by", ["Accuracy", "F1-score", "Precision", "Recall"])

if source.startswith("📁"):
    test_dir = str(shared.loader.BASE_DIR / config.TEST_FOLDER)
else:
    upload = st.file_uploader("Zip file with one folder per class", type="zip")
    if not upload:
        st.info("Upload a **.zip** that has one folder per class (folder name = class name).")
        st.stop()
    test_dir = shared.extract_zip(upload.getvalue())

files, labels, skipped = shared.scan_test_folder(test_dir, tuple(classes), config.MAX_IMAGES_PER_CLASS)
if skipped:
    st.caption("Skipped folders that are not a class name: " + ", ".join(skipped))
if not files:
    st.info(
        f"**No labelled test images found.** Put them in one folder per class:\n\n"
        f"```\n{config.TEST_FOLDER}/\n├── {classes[0]}/   img1.jpg  img2.jpg …\n├── {classes[-1]}/   …\n└── …\n```\n"
        f"Folder names must be the class names: {', '.join(classes)} — or upload a .zip instead."
    )
    st.stop()

# ----------------------------------------------------------------- test every model
y_true = np.asarray(labels)
results = []
progress = st.progress(0.0)
for i, m in enumerate(models):
    progress.progress(i / len(models), text=f"Testing {m.name} on {len(files):,} images ({i + 1}/{len(models)})…")
    probs, ms = shared.evaluate_model(m, files)   # cached: each model is tested only once
    pred = probs.argmax(axis=1)
    results.append({"Model": m.name, "pred": pred, "ms": ms, **shared.score(y_true, pred, len(classes))})
progress.empty()

metrics = ["Accuracy", "F1-score", "Precision", "Recall"]
board = pd.DataFrame([{"Model": r["Model"], **{k: r[k] * 100 for k in metrics}, "ms / image": r["ms"]}
                      for r in results])
board = board.sort_values([rank_by, "ms / image"], ascending=[False, True], ignore_index=True)
board.insert(0, "Rank", [("🥇", "🥈", "🥉")[i] if i < 3 else str(i + 1) for i in range(len(board))])
by_name = {r["Model"]: r for r in results}
order = list(board["Model"])
best = board.iloc[0]
fastest = board.loc[board["ms / image"].idxmin()]
winners = set(board.loc[board[rank_by] >= best[rank_by] - 1e-9, "Model"])

st.markdown('<div class="mcards">' + "".join(
    f'<div class="mcard"><div class="mcard-name">{t}</div><div class="mcard-label">{v}</div>'
    f'<div class="mcard-foot">{n}</div></div>'
    for t, v, n in [("Models compared", len(results), "same test images for all"),
                    ("Test images", f"{len(files):,}", f"{len(set(labels))} classes"),
                    ("Best model", best["Model"], f"{rank_by} {best[rank_by]:.1f}%"),
                    ("Fastest", fastest["Model"], f"{fastest['ms / image']:.1f} ms per image")]
) + "</div>", unsafe_allow_html=True)

st.markdown("#### 🏆 Leaderboard")
st.dataframe(board, hide_index=True, width="stretch", column_config={
    **{k: st.column_config.ProgressColumn(k, format="%.1f%%", min_value=0, max_value=100) if k == rank_by
       else st.column_config.NumberColumn(k, format="%.1f%%") for k in metrics},
    "ms / image": st.column_config.NumberColumn(format="%.1f"),
})

left, right = st.columns(2, gap="medium")
with left.container(border=True):
    st.markdown(f"#### {rank_by} by model")
    st.caption("Higher is better · highlighted = best")
    st.plotly_chart(charts.ranking_bar(board["Model"], board[rank_by], highlight=winners),
                    width="stretch", config=charts.CONFIG)
with right.container(border=True):
    st.markdown("#### Speed (ms per image)")
    st.caption("Lower is better · highlighted = fastest")
    st.plotly_chart(charts.ranking_bar(board["Model"], board["ms / image"], highlight={fastest["Model"]},
                                       percent=False), width="stretch", config=charts.CONFIG)

f1 = np.array([by_name[n]["f1_per_class"] * 100 for n in order])
with st.container(border=True):
    st.markdown("#### 🎯 F1-score per class")
    st.caption("Which model is best for each land type — darker = better")
    st.plotly_chart(charts.heatmap(f1, classes, order, text=[[f"{v:.0f}" for v in row] for row in f1],
                                   hover="<b>%{z:.1f}%</b> F1-score<br>%{y} · %{x}<extra></extra>",
                                   colorbar_title="F1-score"), width="stretch", config=charts.CONFIG)

st.markdown("#### 🧩 Confusion matrix")
st.caption("Rows = true class, columns = what the model predicted. A perfect model only fills the diagonal.")
with st.container(border=True):
    name = st.selectbox("Model", order)
    st.plotly_chart(charts.confusion(by_name[name]["cm"], classes), width="stretch", config=charts.CONFIG)

predictions = pd.DataFrame({"Image": [os.path.relpath(f, test_dir) for f in files],
                            "True class": [classes[i] for i in labels],
                            **{n: [classes[i] for i in by_name[n]["pred"]] for n in order}})
st.download_button("⬇️ Every prediction (CSV)", predictions.to_csv(index=False).encode("utf-8-sig"),
                   "predictions.csv", "text/csv")
