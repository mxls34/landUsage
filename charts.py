"""Plotly charts — one consistent look for every chart in the app."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Validated data-viz palette: one accent + grey to highlight a winner,
# one blue ramp (light = low, dark = high) for every heatmap.
ACCENT = "#2a78d6"
GREY_BAR = "#c3c2b7"
GREY_DOT = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_SCALE = [[i / (len(BLUES) - 1), c] for i, c in enumerate(BLUES)]
FONT = "Poppins, system-ui, -apple-system, 'Segoe UI', sans-serif"

# pass to st.plotly_chart(..., config=CONFIG)
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def _style(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=24, t=40 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(11,11,11,0.10)",
                        font=dict(family=FONT, size=12, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(color=INK_2)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2, size=12))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=INK_2), title_font=dict(color=INK_2, size=12))
    return fig


def _round_bar_ends(fig: go.Figure) -> None:
    try:
        fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    except ValueError:
        pass  # older plotly: square bar ends


def ranking_bar(names, values, highlight, percent: bool = True) -> go.Figure:
    """Horizontal bars, first item on top; highlighted names in the accent colour."""
    names, values = list(names), [float(v) for v in values]
    labels = [f"{v:.1f}%" if percent else f"{v:.2f} ms" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=[ACCENT if n in highlight else GREY_BAR for n in names]),
        text=labels, textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2, size=12),
        hovertemplate="<b>%{text}</b><br>%{y}<extra></extra>",
    ))
    _round_bar_ends(fig)
    height = max(200, 40 + 34 * len(names))
    row_px = (height - 50) / max(len(names), 1)
    fig.update_layout(bargap=max(0.38, 1 - 20 / row_px))   # bars stay ~20px thick
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", ticksuffix="  ")
    if percent:
        fig.update_xaxes(range=[0, 118], tickvals=[0, 25, 50, 75, 100], ticksuffix="%")
    else:
        fig.update_xaxes(range=[0, max(values + [0.001]) * 1.3], ticksuffix=" ms")
    return _style(fig, height=height)


def heatmap(z, x_labels, y_labels, text, hover: str, customdata=None,
            zmax: float = 100, colorbar_title: str = "", tickangle: int = -35) -> go.Figure:
    """Grid of cells on the blue ramp (fixed 0..zmax scale, so colours compare across charts)."""
    fig = go.Figure(go.Heatmap(
        z=z, x=list(x_labels), y=list(y_labels), zmin=0, zmax=zmax, colorscale=BLUE_SCALE,
        xgap=2, ygap=2, text=text, texttemplate="%{text}", textfont=dict(size=11),
        customdata=customdata, hovertemplate=hover,
        colorbar=dict(title=dict(text=colorbar_title, side="right", font=dict(color=INK_2)),
                      thickness=10, outlinewidth=0, ticksuffix="%", tickfont=dict(color=MUTED)),
    ))
    fig.update_xaxes(showgrid=False, tickangle=tickangle, linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(showgrid=False, autorange="reversed", linecolor="rgba(0,0,0,0)", ticksuffix="  ")
    return _style(fig, height=max(240, 110 + 34 * len(list(y_labels))))


def confusion(cm, class_names) -> go.Figure:
    """Rows = true class, columns = predicted. Colour = % of the true class, text = image count."""
    cm = np.asarray(cm, float)
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0) * 100
    text = [[f"{int(v)}" if v else "" for v in row] for row in cm]
    fig = heatmap(pct, class_names, class_names, text, customdata=cm.astype(int),
                  hover="<b>%{z:.1f}%</b> of %{y} → predicted %{x}<br>%{customdata} images<extra></extra>",
                  tickangle=-90)
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="True class")
    return fig
