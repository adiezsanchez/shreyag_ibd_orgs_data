import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import os
    import sys
    from tqdm import tqdm
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px

    from utils_data_analysis import merge_csv_files

    return Path, merge_csv_files, mo, pd, plt, sns


@app.cell
def _(Path, merge_csv_files, pd):
    # Point to the results directory
    results_directory = Path("raw_data/8052_CLDN1.OCLN.ECAD")

    # Extract the experiment name from the results directory (capture the full name, including all extensions)
    experiment_id = results_directory.name

    # Point to the conditions file
    df_conditions = pd.read_csv(f"raw_data/{experiment_id}_conditions.csv")

    # Merge all the csv files into a single dataframe
    df_merged = merge_csv_files(results_directory, df_conditions)
    return (df_merged,)


@app.cell
def _(df_merged, mo):
    group_options = ["1", "2", "3", "4", "5"]
    group_radio = mo.ui.radio(
        options=group_options, value=group_options[0], label="Group to plot"
    )

    x_options = list(df_merged.columns)
    x_radio = mo.ui.radio(
        options=x_options, value=x_options[5], label="X-axis variable"
    )

    y_options = list(df_merged.columns)
    y_radio = mo.ui.radio(
        options=y_options, value=y_options[32], label="Y-axis variable"
    )
    return group_radio, x_radio, y_radio


@app.cell
def _(group_radio, mo, x_radio, y_radio):
    mo.vstack(
        [
            mo.md("## Choose group and variables to plot"),
            mo.hstack([group_radio, x_radio, y_radio], gap=3),
        ]
    )
    return


@app.cell
def _(group_radio, x_radio, y_radio):
    print(group_radio.value)
    print(x_radio.value)
    print(y_radio.value)
    return


@app.cell
def _(df_merged, plt, sns, x_radio, y_radio):
    # Filter out cells not present in organoids
    df_filtered = df_merged[df_merged["organoid"] != 0]

    plt.figure(figsize=(10, 5))
    sns.histplot(
        df_filtered[x_radio.value],
        bins=100,
        color="blue",
        alpha=0.5,
        label=x_radio.value,
    )
    sns.histplot(
        df_filtered[y_radio.value],
        bins=100,
        color="red",
        alpha=0.5,
        label=y_radio.value,
    )
    plt.legend()
    plt.xlabel("Area")
    plt.ylabel("Count")
    plt.title("Distribution of Cell and Membrane Area")

    # Cap the x axis if needed for better visualization
    plt.xlim(0, 20000)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
