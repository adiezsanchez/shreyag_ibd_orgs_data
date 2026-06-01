import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


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
    df_columns = (
        lazy_df.collect_schema().names()
    )  # Efficient equivalent of df.columns
    keywords = ["area", "solidity", "mean_int", "sum_int", "max_mean"]
    exclude_keywords = ["bbox", "convex", "filled", "equivalent"]
    features_to_plot = [
        col
        for col in df_columns
        if any(kw in col for kw in keywords)
        and not any(ex_kw in col for ex_kw in exclude_keywords)
    ]
    return (features_to_plot,)


@app.cell
def _(lazy_df):
    donor_ids = sorted(
        str(d)
        for d in set(
            lazy_df.select("donor_id").collect().get_column("donor_id").to_list()
        )
        if d is not None
    )
    return (donor_ids,)


@app.cell
def _(donor_ids, features_to_plot, mo):
    def make_controls(suffix: str):
        return (
            mo.ui.radio(
                options=donor_ids,
                value=donor_ids[0] if donor_ids else None,
                label=f"Donor ID ({suffix})",
            ),
            mo.ui.radio(
                options=features_to_plot,
                value=features_to_plot[5],
                label=f"X-axis ({suffix})",
            ),
        )

    group_radio_row, x_radio_row = make_controls("row layout")
    return group_radio_row, x_radio_row


@app.cell
def _(donor_ids, mo):
    donor_multiselect = mo.ui.multiselect(
        options=donor_ids,
        value=[donor_ids[0]] if donor_ids else None,
        label="**Donor IDs (`mo.ui.multiselect`)**",
    )
    return (donor_multiselect,)


@app.cell
def _(donor_multiselect, mo):
    mo.vstack(
        [
            mo.md("### Multiselect demo"),
            donor_multiselect,
            mo.md(f"Currently selected: `{donor_multiselect.value}`"),
        ],
        gap=1,
    )
    return


@app.cell
def _(donor_multiselect):
    print("multiselect.value:", donor_multiselect.value)
    print("multiselect.value type:", type(donor_multiselect.value))
    return


@app.cell
def _(donor_ids, mo):
    donor_checkbox_array = mo.ui.array(
        [
            mo.ui.checkbox(label=donor_id, value=(i == 0))
            for i, donor_id in enumerate(donor_ids)
        ]
    )
    return (donor_checkbox_array,)


@app.cell
def _(donor_checkbox_array, donor_ids, mo):
    selected_donors = [
        donor_id
        for donor_id, checked in zip(donor_ids, donor_checkbox_array.value)
        if checked
    ]

    mo.vstack(
        [
            mo.md("### Checkbox array demo (`mo.ui.array`)"),
            donor_checkbox_array,
            mo.md(f"Currently selected: `{selected_donors}`"),
        ],
        gap=1,
    )
    return (selected_donors,)


@app.cell
def _(donor_checkbox_array, selected_donors):
    print("array.value (per-checkbox booleans):", donor_checkbox_array.value)
    print("selected donor IDs from array:", selected_donors)
    return


@app.cell
def _(lazy_df, pl):
    # Filter out cells not present in organoids (shared by all layout demos)
    df_collected = lazy_df.filter(pl.col("organoid") != 0).collect()
    return (df_collected,)


@app.cell
def _(plt, sns):
    def build_histogram(df_collected, x_var):
        plt.figure(figsize=(8, 4))
        sns.histplot(
            df_collected[x_var].to_numpy(),
            bins=100,
            color="blue",
            alpha=0.5,
            label=x_var,
        )
        plt.legend()
        plt.xlabel(x_var)
        plt.ylabel("Count")
        plt.title(f"Distribution of {x_var}")
        return plt.gcf()

    return (build_histogram,)


@app.cell
def _(build_histogram, df_collected, group_radio_row, mo, x_radio_row):
    fig_row = build_histogram(df_collected, x_radio_row.value)
    mo.vstack(
        [
            mo.md(
                "Controls and plot in one row; plot sits to the right of the X-axis radio."
            ),
            mo.hstack(
                [group_radio_row, x_radio_row, mo.mpl.interactive(fig_row)],
                gap=3,
                align="start",
                widths=[1, 2, 5],
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
