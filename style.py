"""🧊 Liquid-glass look + small HTML building blocks."""
from html import escape

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family: "Poppins", system-ui, -apple-system, "Segoe UI", sans-serif; }

/* Background: soft beige with colourful blurred blobs */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% 18%, rgba(139, 92, 246, 0.35) 0, transparent 32%),
        radial-gradient(circle at 88% 22%, rgba(45, 212, 191, 0.30) 0, transparent 30%),
        radial-gradient(circle at 80% 85%, rgba(244, 114, 182, 0.28) 0, transparent 32%),
        radial-gradient(circle at 15% 90%, rgba(125, 211, 252, 0.30) 0, transparent 30%),
        linear-gradient(135deg, #efe9e3 0%, #e4e2e6 50%, #dcdde3 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }

/* Main glass panel */
.block-container {
    max-width: 980px;
    background: rgba(255, 255, 255, 0.28);
    backdrop-filter: blur(22px) saturate(160%);
    -webkit-backdrop-filter: blur(22px) saturate(160%);
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: 32px;
    box-shadow: 0 20px 50px rgba(80, 70, 110, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.8);
    padding: 2.2rem 2.2rem 2.6rem !important;
    margin: 2.5rem auto;
}
@media (max-width: 640px) {
    .block-container { padding: 1.2rem 1rem !important; margin: 1rem auto; border-radius: 24px; }
}
h1, h2, h3, h4, p, label { color: #1f1d2b; }

/* Glass cards */
.glass-card, .mcard {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.58), rgba(255, 255, 255, 0.22));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.75);
    box-shadow: 0 10px 30px rgba(80, 70, 110, 0.13), inset 0 1px 0 rgba(255, 255, 255, 0.9);
}
.glass-card { border-radius: 24px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; }
.glass-card.rainbow {
    background: linear-gradient(135deg, rgba(251, 207, 232, 0.65), rgba(196, 181, 253, 0.55) 45%, rgba(153, 246, 228, 0.55));
}
.glass-card .glass-title { font-size: 2rem !important; font-weight: 700; margin: 0; line-height: 1.3; }
.glass-card .glass-sub { color: #5b5870; margin: 0.25rem 0 0 0; }
.glass-card .big-label { font-size: 2rem !important; font-weight: 700; margin: 0.45rem 0 0.1rem 0; }
.glass-card .soft { color: #5b5870; margin: 0.1rem 0; }
.pill {
    display: inline-block; padding: 0.22rem 0.8rem; border-radius: 999px;
    background: rgba(255, 255, 255, 0.75); border: 1px solid rgba(255, 255, 255, 0.95);
    font-size: 0.78rem; color: #4c4766; margin-right: 0.35rem;
}

/* One card per model */
.mcards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; margin: 0.4rem 0 1.2rem; }
.mcard { border-radius: 22px; padding: 1rem 1.2rem; min-width: 0; }
.mcard-name { font-size: 0.85rem; color: #5b5870; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mcard-label { font-size: 1.25rem; font-weight: 600; color: #1f1d2b; margin: 0.2rem 0 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mcard .tag { font-size: 0.72rem; padding: 0.05rem 0.55rem; border-radius: 999px; background: rgba(255, 255, 255, 0.85); color: #52514e; border: 1px solid rgba(11, 11, 11, 0.08); margin-left: 0.3rem; }
.row { display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.82rem; color: #52514e; margin-top: 0.45rem; }
.row span:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.meter { height: 6px; border-radius: 999px; background: #cde2fb; overflow: hidden; margin-top: 0.2rem; }
.meter > span { display: block; height: 100%; border-radius: 999px; background: #2a78d6; }
.meter.soft > span { background: #9ec5f4; }
.mcard-foot { font-size: 0.75rem; color: #898781; margin-top: 0.6rem; }

/* Bordered containers -> glass */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.30);
    border: 1px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 24px !important;
}

/* Buttons: purple pill */
.stButton > button {
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: 999px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    box-shadow: 0 10px 22px rgba(124, 58, 237, 0.30), inset 0 2px 0 rgba(255, 255, 255, 0.45);
    transition: all 0.2s ease;
}
.stButton > button p { color: #ffffff !important; font-size: 1.05rem; }
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(124, 58, 237, 0.42), inset 0 2px 0 rgba(255, 255, 255, 0.5);
    border-color: rgba(255, 255, 255, 0.9);
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.45);
    border: 1.5px dashed rgba(124, 58, 237, 0.45);
    border-radius: 22px;
}
[data-testid="stImage"] img { border-radius: 20px; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12); }
[data-testid="stAlert"] { border-radius: 18px; }
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="glass-card"><p class="glass-title">{escape(title)}</p>'
                f'<p class="glass-sub">{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def answer_card(label: str, pills: list, lines: list) -> None:
    st.markdown(
        '<div class="glass-card rainbow">'
        + "".join(f'<span class="pill">{escape(p)}</span>' for p in pills)
        + f'<p class="big-label">{escape(label)}</p>'
        + "".join(f'<p class="soft">{escape(t)}</p>' for t in lines)
        + "</div>",
        unsafe_allow_html=True,
    )


def model_cards(cards: list) -> None:
    """cards: dicts with name, arch, label, top (list of (class, percent)), ms, agrees."""
    html = ""
    for c in cards:
        rows = "".join(
            f'<div class="row"><span>{escape(name)}</span><span>{pct:.1f}%</span></div>'
            f'<div class="meter{"" if i == 0 else " soft"}"><span style="width:{pct:.1f}%"></span></div>'
            for i, (name, pct) in enumerate(c["top"])
        )
        tag = "" if c["agrees"] else '<span class="tag">differs</span>'
        html += (f'<div class="mcard"><div class="mcard-name" title="{escape(c["name"])}">{escape(c["name"])}'
                 f'<span class="tag">{escape(c["arch"])}</span>{tag}</div>'
                 f'<div class="mcard-label">{escape(c["label"])}</div>{rows}'
                 f'<div class="mcard-foot">{c["ms"]:.0f} ms</div></div>')
    st.markdown(f'<div class="mcards">{html}</div>', unsafe_allow_html=True)
