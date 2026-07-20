"""
visualizer.py
Generates Plotly charts for EDA results.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BG  = "#0d0d0d"
_PLT = "#111111"
_TPL = "plotly_dark"

def _style(fig):
    """Apply consistent dark background to any figure."""
    fig.update_layout(paper_bgcolor=_BG, plot_bgcolor=_PLT)
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure | None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None
    fig = px.bar(
        x=missing.index, y=missing.values,
        labels={"x": "Column", "y": "Missing Count"},
        title="Missing Values per Column",
        color=missing.values, color_continuous_scale="reds",
        template=_TPL,
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> go.Figure | None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return None
    cols_per_row = 3
    n_rows = (len(num_cols) + cols_per_row - 1) // cols_per_row
    fig = make_subplots(rows=n_rows, cols=cols_per_row, subplot_titles=num_cols)
    for i, col in enumerate(num_cols):
        row = i // cols_per_row + 1
        col_pos = i % cols_per_row + 1
        fig.add_trace(
            go.Histogram(x=df[col], name=col, nbinsx=30, marker_color="#6366f1"),
            row=row, col=col_pos,
        )
    fig.update_layout(
        title="Numeric Column Distributions",
        showlegend=False, height=300 * n_rows,
        template=_TPL,
    )
    return _style(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Heatmap", template=_TPL)
    return _style(fig)


def plot_boxplots(df: pd.DataFrame) -> go.Figure | None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return None
    fig = go.Figure()
    for col in num_cols:
        fig.add_trace(go.Box(y=df[col], name=col, boxpoints="outliers"))
    fig.update_layout(
        title="Box Plots – Spread & Outliers",
        template=_TPL, showlegend=False,
    )
    return _style(fig)


def plot_categorical_bars(df: pd.DataFrame, top_n: int = 10) -> list[go.Figure]:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    figs = []
    for col in cat_cols:
        vc = df[col].value_counts().head(top_n).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(
            vc, x=col, y="count",
            title=f"Top {top_n} Values – {col}",
            color="count", color_continuous_scale="viridis",
            template=_TPL,
        )
        fig.update_layout(coloraxis_showscale=False)
        figs.append(_style(fig))
    return figs


def plot_time_series(df: pd.DataFrame) -> list[go.Figure]:
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    figs = []
    for dc in date_cols:
        for nc in num_cols[:4]:
            tmp = df[[dc, nc]].dropna().sort_values(dc)
            fig = px.line(tmp, x=dc, y=nc, title=f"{nc} over {dc}", template=_TPL)
            figs.append(_style(fig))
    return figs


def plot_anomalies(df: pd.DataFrame, anomaly_col: str,
                   anomaly_labels: pd.Series) -> go.Figure | None:
    if anomaly_col not in df.columns:
        return None
    tmp = df[[anomaly_col]].copy()
    tmp["anomaly"] = anomaly_labels.values
    tmp["index"]   = range(len(tmp))
    normal  = tmp[tmp["anomaly"] ==  1]
    outlier = tmp[tmp["anomaly"] == -1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["index"], y=normal[anomaly_col],
        mode="markers", name="Normal",
        marker=dict(color="#4ade80", size=5)
    ))
    fig.add_trace(go.Scatter(
        x=outlier["index"], y=outlier[anomaly_col],
        mode="markers", name="Anomaly",
        marker=dict(color="#f87171", size=9, symbol="x")
    ))
    fig.update_layout(
        title=f"Anomaly Detection – {anomaly_col}",
        xaxis_title="Row Index", yaxis_title=anomaly_col,
        template=_TPL,
    )
    return _style(fig)


def plot_pairplot_sample(df: pd.DataFrame, max_cols: int = 4) -> go.Figure | None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:max_cols]
    if len(num_cols) < 2:
        return None
    fig = px.scatter_matrix(
        df[num_cols].dropna(),
        dimensions=num_cols,
        title="Pair Plot (Scatter Matrix)",
        template=_TPL,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
    return _style(fig)
