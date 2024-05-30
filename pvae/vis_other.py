import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_posterior_means_for_df(df_means: pd.DataFrame, axis_range) -> go.Figure:
    fig = px.scatter(
        df_means,
        x="z0",
        y="z1",
        color="class_label",
        title="Posterior Means",
        category_orders={"class_label": sorted(df_means["class_label"].unique())},
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        xaxis_title="z0",
        yaxis_title="z1",
        xaxis_range=axis_range,
        yaxis_range=axis_range,
        width=800,
        height=800,
    )
    return fig
