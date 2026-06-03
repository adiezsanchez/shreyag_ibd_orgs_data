import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import plotly.express as px
    import polars as pl
    import seaborn as sns

    from utils_data_analysis import (
        build_condition_order_by_group,
        condition_hue_order,
        get_unique_values,
        merge_csv_files,
    )
    from utils_data_plotting import build_histogram

    return (
        Path,
        build_condition_order_by_group,
        build_histogram,
        condition_hue_order,
        get_unique_values,
        merge_csv_files,
        mo,
        pl,
        px,
        sns,
    )


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
    return donor_ids, group_ids


@app.cell
def _(donor_ids, features_to_plot, group_ids, mo):
    # Generate UI elements for the different plots
    donor_checkbox_array = mo.ui.array(
        [mo.ui.checkbox(label=donor_id, value=(i == 0)) for i, donor_id in enumerate(donor_ids)],
        label="Donor ID (multiselect)",
    )

    group_options = ["None", *group_ids]
    group_radio = mo.ui.radio(
        options=group_options,
        value="None",
        label="Treatment groups",
    )

    x_radio = mo.ui.radio(options=features_to_plot, value=features_to_plot[5], label="X-axis")
    y_radio = mo.ui.radio(options=features_to_plot, value=features_to_plot[6], label="Y-axis")

    # Generate data aggregation strategy UI element
    aggregation_strategies = ["single_cell", "organoid", "well"]

    aggregation_radio = mo.ui.radio(
        options=aggregation_strategies,
        value="well",
        label="Aggregate data (average) by:",
    )
    return (
        aggregation_radio,
        donor_checkbox_array,
        group_radio,
        x_radio,
        y_radio,
    )


@app.cell
def _(build_condition_order_by_group, lazy_df, pl):
    # Filter out cells not present in organoids
    df_collected = lazy_df.filter(pl.col("organoid") != 0).collect()
    condition_order_by_group = build_condition_order_by_group(df_collected)
    return condition_order_by_group, df_collected


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
    _selected_donors = [
        donor_id
        for donor_id, checked in zip(donor_ids, donor_checkbox_array.value, strict=True)
        if checked
    ]

    # Filter by DataFrame by donor
    if _selected_donors:
        _df_plot = df_collected.filter(pl.col("donor_id").cast(pl.Utf8).is_in(_selected_donors))
        _hue_var = "donor_id"
    else:  # if no donor selected plot all
        _df_plot = df_collected
        _hue_var = None

    fig_row = build_histogram(
        df=_df_plot,
        x_var=x_radio.value,
        hue_var=_hue_var,
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
                widths=[1, 1, 2],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    aggregation_radio,
    condition_hue_order,
    condition_order_by_group,
    df_collected,
    donor_checkbox_array,
    donor_ids,
    group_radio,
    pl,
):
    selected_donors = [
        donor_id
        for donor_id, checked in zip(donor_ids, donor_checkbox_array.value, strict=True)
        if checked
    ]

    if selected_donors:
        df_plot = df_collected.filter(pl.col("donor_id").cast(pl.Utf8).is_in(selected_donors))
    else:
        df_plot = df_collected

    selected_group = group_radio.value
    if selected_group and selected_group != "None":
        df_plot = df_plot.filter(pl.col("group_number").cast(pl.Utf8) == selected_group)

    if aggregation_radio.value == "single_cell":
        df_plot_aggregated = df_plot
    elif aggregation_radio.value == "well":
        df_plot_aggregated = (
            df_plot.group_by(["donor_id", "well_id", "condition"])
            .mean()
            .drop(["organoid", "multiposition_id", "label"], strict=False)
        )
    else:
        df_plot_with_organoid_id = df_plot.with_columns(
            pl.concat_str(
                [
                    pl.col("multiposition_id").cast(pl.Utf8),
                    pl.col("organoid").cast(pl.Utf8),
                ],
                separator="_",
            ).alias("unique_organoid_id")
        )
        df_plot_aggregated = (
            df_plot_with_organoid_id.group_by(
                ["well_id", "unique_organoid_id", "condition", "donor_id"]
            )
            .mean()
            .drop(["organoid", "multiposition_id", "label"], strict=False)
        )

    correlation_dataframe = df_plot_aggregated.to_pandas()
    present_conditions = set(correlation_dataframe["condition"].dropna().astype(str))
    hue_order = condition_hue_order(
        condition_order_by_group,
        selected_group=selected_group,
        present_conditions=present_conditions,
    )
    return correlation_dataframe, hue_order, selected_group


@app.cell
def _(
    aggregation_radio,
    correlation_dataframe,
    donor_checkbox_array,
    group_radio,
    hue_order,
    mo,
    px,
    x_radio,
    y_radio,
):
    _x_col = x_radio.value
    _y_col = y_radio.value
    dataframe = correlation_dataframe[["condition", "donor_id", _x_col, _y_col]].dropna(
        subset=[_x_col, _y_col]
    )
    _scatter_kws = {
        "x": _x_col,
        "y": _y_col,
        "color": "condition",
        "hover_data": ["donor_id"],
        "marginal_x": "violin",
        "marginal_y": "violin",
        "opacity": 0.4,
        "title": f"Data aggregation: {aggregation_radio.value}",
    }
    if hue_order:
        _scatter_kws["category_orders"] = {"condition": hue_order}
    _fig = px.scatter(dataframe, **_scatter_kws)
    _fig.update_layout(
        height=700,
        width=1000,
        legend_title_text="Treatment",
    )

    mo.vstack(
        [
            mo.md("Feature correlation plots (Plotly)"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    aggregation_radio,
                    group_radio,
                    x_radio,
                    y_radio,
                    mo.ui.plotly(_fig),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 1, 1, 1, 10],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    aggregation_radio,
    correlation_dataframe,
    donor_checkbox_array,
    group_radio,
    hue_order,
    mo,
    sns,
    x_radio,
    y_radio,
):
    jointplot_kws = dict(
        x=x_radio.value,
        y=y_radio.value,
        data=correlation_dataframe,
        hue="condition",
        joint_kws=dict(alpha=0.4, s=20),
    )
    if hue_order:
        jointplot_kws["hue_order"] = hue_order
    g = sns.jointplot(**jointplot_kws)
    g.figure.set_size_inches(12, 7)
    # Move legend outside the main axes so it does not cover data points
    sns.move_legend(
        g.ax_joint,
        "upper left",
        bbox_to_anchor=(1.22, 1),
        borderaxespad=0,
        title="Treatment",
    )
    g.figure.suptitle(f"Data aggregation: {aggregation_radio.value}", fontsize=14)
    g.figure.subplots_adjust(top=0.93, right=0.74)

    mo.vstack(
        [
            mo.md("Feature correlation plots (Seaborn)"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    aggregation_radio,
                    group_radio,
                    x_radio,
                    y_radio,
                    mo.mpl.interactive(g.figure),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 1, 1, 1, 10],
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
