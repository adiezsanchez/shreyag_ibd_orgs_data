import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from utils_data_analysis import get_unique_values, merge_csv_files
    from utils_data_plotting import build_histogram

    return Path, build_histogram, get_unique_values, merge_csv_files, mo, pl


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
    # Aggregate all experiments into a single lazy DataFrame
    lazy_df = pl.scan_parquet("processed_data/*.parquet")
    return (lazy_df,)


@app.cell
def _(lazy_df):
    # Extract column names from DataFrame
    df_columns = lazy_df.collect_schema().names()

    # Remove metadata and unwanted features from features to plot
    keywords = ["area", "solidity", "mean_int", "sum_int", "max_mean"]
    exclude_keywords = ["bbox", "convex", "filled", "equivalent"]
    features_to_plot = [
        col
        for col in df_columns
        if any(kw in col for kw in keywords) and not any(ex_kw in col for ex_kw in exclude_keywords)
    ]
    return (features_to_plot,)


@app.cell
def _(get_unique_values, lazy_df):
    # Dinamically generate options from metadata columns
    # Generate options for donor_id
    donor_ids = get_unique_values(df=lazy_df, column_name="donor_id")

    # Generate options for group_id (groups of variables to plot)
    group_ids = get_unique_values(df=lazy_df, column_name="group_number")

    # Generate options for treatment_id
    treatment_ids = get_unique_values(df=lazy_df, column_name="condition")
    return donor_ids, group_ids, treatment_ids


@app.cell
def _(donor_ids, features_to_plot, group_ids, mo):
    # Generate UI elements for the different plots
    donor_checkbox_array = mo.ui.array(
        [mo.ui.checkbox(label=donor_id, value=(i == 0)) for i, donor_id in enumerate(donor_ids)],
        label="Donor ID (multiselect)",
    )

    group_radio = mo.ui.radio(
        options=group_ids,
        value=group_ids[0] if group_ids else None,
        label="Group plots",
    )

    x_radio = mo.ui.radio(options=features_to_plot, value=features_to_plot[5], label="X-axis")
    return donor_checkbox_array, group_radio, x_radio


@app.cell
def _(lazy_df, pl):
    # Filter out cells not present in organoids
    df_collected = lazy_df.filter(pl.col("organoid") != 0).collect()
    return (df_collected,)


@app.cell
def _(
    build_histogram,
    df_collected,
    donor_checkbox_array,
    donor_ids,
    mo,
    pl,
    x_radio,
):
    # Collect selected donors from the checkbox_array
    selected_donors = [
        donor_id
        for donor_id, checked in zip(donor_ids, donor_checkbox_array.value, strict=True)
        if checked
    ]

    # Filter by DataFrame by donor
    if selected_donors:
        df_plot = df_collected.filter(pl.col("donor_id").cast(pl.Utf8).is_in(selected_donors))
        hue_var = "donor_id"
    else:  # if no donor selected plot all
        df_plot = df_collected
        hue_var = None

    fig_row = build_histogram(
        df=df_plot,
        x_var=x_radio.value,
        hue_var=hue_var,
        cmap_name="viridis",
    )

    mo.vstack(
        [
            mo.md("Single cell feature value distribution"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    x_radio,
                    mo.mpl.interactive(fig_row),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 2, 5],
            ),
        ],
        gap=1,
    )
    return (selected_donors,)


@app.cell
def _(donor_checkbox_array, selected_donors):
    print("array.value (per-checkbox booleans):", donor_checkbox_array.value)
    print("selected donor IDs from array:", selected_donors)
    return


if __name__ == "__main__":
    app.run()
