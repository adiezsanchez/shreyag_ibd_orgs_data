import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    from utils_data_analysis import merge_csv_files

    return Path, merge_csv_files, mo, pl, plt, sns


@app.cell
def _(Path, merge_csv_files, pl):
    # Process all experiments in raw_data/
    # Aggregate per well_id .csvs into a experiment_id .parquet file
    raw_data_dir = Path("raw_data")

    for results_directory in raw_data_dir.iterdir():
        if results_directory.is_dir():
            # Extract the experiment name from the results directory
            experiment_id = results_directory.name

            # Point to the conditions file
            df_conditions_path = raw_data_dir / f"{experiment_id}_conditions.csv"
            if not df_conditions_path.exists():
                print(f"Conditions file not found for {experiment_id}, skipping.")
                continue
            df_conditions = pl.read_csv(str(df_conditions_path))

            # Merge all the csv files into a single dataframe
            merge_csv_files(results_directory, df_conditions)
    return


@app.cell
def _(pl):
    lazy_df = pl.scan_parquet("processed_data/*.parquet")
    return (lazy_df,)


@app.cell
def _(lazy_df):
    lazy_df.collect_schema().names()  # Efficient equivalent of df.columns
    return


@app.cell
def _(df_merged, mo):
    group_options = ["1", "2", "3", "4", "5"]
    group_radio = mo.ui.radio(options=group_options, value=group_options[0], label="Group to plot")

    x_options = list(df_merged.columns)
    x_radio = mo.ui.radio(options=x_options, value=x_options[5], label="X-axis variable")

    y_options = list(df_merged.columns)
    y_radio = mo.ui.radio(options=y_options, value=y_options[32], label="Y-axis variable")
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
def _(df_merged, mo, plt, sns, x_radio, y_radio):
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

    mo.mpl.interactive(plt.gcf())
    return


if __name__ == "__main__":
    app.run()
