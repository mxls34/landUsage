"""📊 Dashboard — compare every model's answer for the pictures you predicted."""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import shared
import style

style.header("📊 Dashboard", "Compare the answer of every model for the pictures you predicted on the Predict page.")

history = st.session_state.get("history", [])
if not history:
    st.info("No prediction yet — go to **Predict**, upload a picture and press **✨ Predict**. "
            "The answers of every model will be compared here.")
    st.page_link("views/predict.py", label="Go to Predict", icon="🔍")
    st.stop()

# "Rank by" list: label -> (column, lowest first?)
RANK_BY = {
    "Confidence (high → low)": ("Confidence", False),
    "Agreement with the final answer": ("Agreement", False),
    "Speed (fast → slow)": ("ms", True),
    "Model name (A → Z)": ("Model", True),
}
MEDALS = ("🥇", "🥈", "🥉")


def ranked(df: pd.DataFrame, choice: str) -> pd.DataFrame:
    column, ascending = RANK_BY[choice]
    df = df.sort_values([column, "Model"], ascending=[ascending, True], ignore_index=True)
    df.insert(0, "Rank", [MEDALS[i] if i < 3 else str(i + 1) for i in range(len(df))])
    return df


c_pic, c_rank = st.columns([2, 1])
pick = c_pic.selectbox("Picture", range(len(history)), index=len(history) - 1,
                       format_func=lambda i: f"{i + 1}. {history[i]['file']}")
rank_by = c_rank.selectbox("Rank by", list(RANK_BY))
entry = history[pick]

# ----------------------------------------------------------------- this picture
df = pd.DataFrame(entry["rows"])
df["Agreement"] = (df["Answer"] == entry["final"]) * 100.0
df = ranked(df, rank_by)
n_agree = int((df["Agreement"] > 0).sum())

c_img, c_answer = st.columns([1, 2], gap="medium")
c_img.image(entry["thumb"], caption=entry["file"], width="stretch")
with c_answer:
    style.answer_card(shared.label(entry["final"]), ["Final answer (majority vote)",
                                                    f"{n_agree} of {len(df)} models agree"],
                      [f"Most confident: {df.loc[df['Confidence'].idxmax(), 'Model']} "
                       f"({df['Confidence'].max():.1f}%)",
                       f"Fastest: {df.loc[df['ms'].idxmin(), 'Model']} ({df['ms'].min():.0f} ms)"])

st.markdown(f"#### 🏆 Models ranked by {rank_by.split(' (')[0].lower()}")
st.dataframe(
    df.assign(Answer=df["Answer"].map(shared.label), Agrees=np.where(df["Agreement"] > 0, "✅", "❌"))
      [["Rank", "Model", "Architecture", "Answer", "Confidence", "Agrees", "ms"]],
    hide_index=True, width="stretch",
    column_config={
        "Confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100),
        "Agrees": st.column_config.TextColumn("Agrees with final answer"),
        "ms": st.column_config.NumberColumn("Time (ms)", format="%.0f"),
    },
)

left, right = st.columns(2, gap="medium")
with left.container(border=True):
    st.markdown("#### Confidence")
    st.caption("How sure each model is about its answer · highlighted = most sure")
    st.plotly_chart(charts.ranking_bar(df["Model"], df["Confidence"],
                                       highlight={df.loc[df["Confidence"].idxmax(), "Model"]}),
                    width="stretch", config=charts.CONFIG)
with right.container(border=True):
    st.markdown("#### Speed (ms)")
    st.caption("Time to answer this picture · highlighted = fastest")
    st.plotly_chart(charts.ranking_bar(df["Model"], df["ms"], highlight={df.loc[df["ms"].idxmin(), "Model"]},
                                       percent=False), width="stretch", config=charts.CONFIG)

classes = entry["rows"][0]["classes"]
if all(r["classes"] == classes for r in entry["rows"]):
    probs = {r["Model"]: r["probs"] for r in entry["rows"]}
    z = np.array([probs[m] for m in df["Model"]]) * 100
    with st.container(border=True):
        st.markdown("#### 📊 Probability of every class")
        st.caption("One row per model (same order as the ranking) — darker = more sure")
        st.plotly_chart(charts.heatmap(z, classes, list(df["Model"]),
                                       text=[[f"{v:.0f}%" if v >= 1 else "" for v in row] for row in z],
                                       hover="<b>%{z:.1f}%</b> %{x}<br>%{y}<extra></extra>",
                                       colorbar_title="Probability"),
                        width="stretch", config=charts.CONFIG)

# ----------------------------------------------------------------- every picture so far
if len(history) > 1:
    st.markdown(f"#### 🗂️ All {len(history)} pictures — models ranked by {rank_by.split(' (')[0].lower()}")
    rows = [dict(r, Agreement=100.0 * (r["Answer"] == h["final"])) for h in history for r in h["rows"]]
    summary = (pd.DataFrame(rows).groupby(["Model", "Architecture"], as_index=False)
               .agg(Pictures=("Answer", "size"), Confidence=("Confidence", "mean"),
                    Agreement=("Agreement", "mean"), ms=("ms", "mean")))
    st.dataframe(
        ranked(summary, rank_by)[["Rank", "Model", "Architecture", "Pictures", "Confidence", "Agreement", "ms"]],
        hide_index=True, width="stretch",
        column_config={
            "Confidence": st.column_config.ProgressColumn("Avg confidence", format="%.1f%%",
                                                          min_value=0, max_value=100),
            "Agreement": st.column_config.ProgressColumn("Agrees with final answer", format="%.0f%%",
                                                         min_value=0, max_value=100),
            "ms": st.column_config.NumberColumn("Avg time (ms)", format="%.0f"),
        },
    )
    answers = pd.DataFrame([{"Picture": f"{i + 1}. {h['file']}", "Final answer": shared.label(h["final"]),
                             **{r["Model"]: shared.label(r["Answer"]) for r in h["rows"]}}
                            for i, h in enumerate(history)])
    with st.expander("Every answer, picture by picture"):
        st.dataframe(answers, hide_index=True, width="stretch")

if st.button("🗑️ Clear the predictions"):
    st.session_state.history = []
    st.rerun()
