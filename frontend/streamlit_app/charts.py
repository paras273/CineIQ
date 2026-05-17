from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def genre_radar(genres: Dict[str, int]):
    if not genres:
        return go.Figure()
    labels = list(genres.keys())
    values = list(genres.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return fig


def decade_bar(decades: Dict[str, int]):
    if not decades:
        return px.bar()
    df = pd.DataFrame({"decade": list(decades.keys()), "count": list(decades.values())})
    df = df.sort_values("decade")
    return px.bar(df, x="decade", y="count")


def top_bar(items: Dict[str, int], title: str):
    if not items:
        return px.bar()
    df = pd.DataFrame({"name": list(items.keys()), "count": list(items.values())})
    df = df.sort_values("count", ascending=False).head(10)
    return px.bar(df, x="name", y="count", title=title)
