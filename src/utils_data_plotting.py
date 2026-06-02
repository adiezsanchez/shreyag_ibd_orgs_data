import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns


def _to_pandas(df):
    """
    Convert supported dataframe inputs to a pandas DataFrame.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input dataframe to convert.

    Returns
    -------
    pd.DataFrame
        The converted pandas dataframe.

    Raises
    ------
    TypeError
        If the input type is neither pandas nor polars DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    msg = f"Unsupported dataframe type: {type(df)!r}"
    raise TypeError(msg)


def build_histogram(df, x_var, hue_var=None, cmap_name="viridis", bins=100):
    """
    Plot a histogram for one feature with optional hue separation.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input dataframe containing plotting columns.
    x_var : str
        Feature name shown on the x-axis.
    hue_var : str | None, optional
        Optional column name used to color histogram groups.
    cmap_name : str, default="viridis"
        Name of matplotlib colormap used when hue grouping is enabled.
    bins : int, default=100
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the histogram plot.
    """
    plt.figure(figsize=(8, 4))
    plot_df = _to_pandas(df)

    palette = None
    if hue_var is not None:
        hue_levels = sorted(plot_df[hue_var].dropna().unique(), key=str)
        cmap = plt.get_cmap(cmap_name)
        palette = {
            level: cmap(i / max(len(hue_levels) - 1, 1)) for i, level in enumerate(hue_levels)
        }

    sns.histplot(
        data=plot_df,
        x=x_var,
        hue=hue_var,
        bins=bins,
        alpha=0.5,
        palette=palette,
        common_norm=False,
    )

    plt.xlabel(x_var)
    plt.ylabel("Count")
    plt.title(f"Distribution of {x_var}")
    return plt.gcf()


def plot_plate_view(df, column_name, title, label, save_dir, fmt=0, display=True, cmap="magma"):
    """
    Plot a 96-well plate heatmap from well-level values.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input dataframe containing ``well_id`` and a value column.
    column_name : str
        Name of the numeric value column used for heatmap intensity.
    title : str
        Plot title.
    label : str
        Colorbar label text.
    save_dir : str
        Unused compatibility parameter retained for call-site stability.
    fmt : int, default=0
        Numeric precision used for cell annotations.
    display : bool, default=True
        Whether to display the figure via ``plt.show()``.
    cmap : str, default="magma"
        Colormap used by the heatmap.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plate heatmap.
    """
    pl_df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(_to_pandas(df))

    plate_long = (
        pl_df.with_columns(
            [
                pl.col("well_id").cast(pl.Utf8).str.extract(r"^([A-H])", 1).alias("row"),
                pl.col("well_id")
                .cast(pl.Utf8)
                .str.extract(r"^[A-H](\d{1,2})$", 1)
                .cast(pl.Int64)
                .alias("col"),
            ]
        )
        .filter(pl.col("row").is_not_null() & pl.col("col").is_not_null())
        .select(["row", "col", column_name])
    )
    plate_matrix = plate_long.to_pandas().pivot(index="row", columns="col", values=column_name)

    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    plate_matrix = plate_matrix.reindex(index=rows, columns=cols)

    # --- Plot heatmap ---
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        plate_matrix,
        cmap=cmap,  # or "coolwarm", "magma" etc.
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": label},
        annot=True,
        fmt=f".{fmt}f",
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Column")
    plt.ylabel("Row")

    # Rotate row (y-axis) labels 90° to the right
    ax.set_yticklabels(ax.get_yticklabels(), rotation=-90, va="center")

    fig = plt.gcf()

    if display:
        plt.show()

    return fig
