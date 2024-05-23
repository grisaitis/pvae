from pathlib import Path
from typing import Tuple
import pandas as pd
import plotly.express as px
import torch


def plot_posterior_means_for_df(df_means: pd.DataFrame, axis_range):
    fig = px.Scatter(
        df_means,
        x="z0",
        y="z1",
        color="class_label",
        title="Posterior Means",
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        xaxis_title="z0",
        yaxis_title="z1",
        xaxis_range=axis_range,
        yaxis_range=axis_range,
    )
    return fig
