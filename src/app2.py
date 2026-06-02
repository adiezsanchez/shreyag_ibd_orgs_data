import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from collections import OrderedDict
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    from utils_data_analysis import (
        build_condition_order_by_group,
        condition_hue_order,
        get_unique_values,
        merge_csv_files,
    )
    from utils_data_plotting import plot_plate_view

    return (
        OrderedDict,
        Path,
        build_condition_order_by_group,
        condition_hue_order,
        get_unique_values,
        merge_csv_files,
        mo,
        pl,
        plot_plate_view,
        plt,
        sns,
    )


@app.cell
def _(Path, merge_csv_files, pl):
    raw_data_dir = Path("raw_data")

    for results_directory in raw_data_dir.iterdir():
        if results_directory.is_dir():
            experiment_id = results_directory.name

            df_conditions_path = raw_data_dir / f"{experiment_id}_conditions.csv"
            if not df_conditions_path.exists():
                print(f"Conditions file not found for {experiment_id}, skipping.")
                continue
            df_conditions = pl.read_csv(str(df_conditions_path))

            merge_csv_files(results_directory, df_conditions)
    return


@app.cell
def _(pl):
    lazy_df = pl.scan_parquet("processed_data/*.parquet")
    return (lazy_df,)


@app.cell
def _(lazy_df):
    df_columns = lazy_df.collect_schema().names()

    signal_intensity_candidates = [
        "membrane_Occludin_RFP_mean_int",
        "membrane_Occludin_RFP_sum_int",
        "membrane_Occludin_RFP_max_mean_ratio",
        "cell_Occludin_RFP_mean_int",
        "cell_Occludin_RFP_sum_int",
        "cell_Occludin_RFP_max_mean_ratio",
        "membrane_Claudin-1_AF750_mean_int",
        "membrane_Claudin-1_AF750_sum_int",
        "membrane_Claudin-1_AF750_max_mean_ratio",
        "cell_Claudin-1_AF750_mean_int",
        "cell_Claudin-1_AF750_sum_int",
        "cell_Claudin-1_AF750_max_mean_ratio",
        "membrane_CellMask_AF647_mean_int",
        "membrane_CellMask_AF647_sum_int",
        "membrane_CellMask_AF647_max_mean_ratio",
        "cell_CellMask_AF647_mean_int",
        "cell_CellMask_AF647_sum_int",
        "cell_CellMask_AF647_max_mean_ratio",
        "nuclei_DAPI_mean_int",
        "nuclei_DAPI_sum_int",
        "nuclei_DAPI_max_mean_ratio",
        "membrane_E-Cadherin_FITC_mean_int",
        "membrane_E-Cadherin_FITC_sum_int",
        "membrane_E-Cadherin_FITC_max_mean_ratio",
        "cell_E-Cadherin_FITC_mean_int",
        "cell_E-Cadherin_FITC_sum_int",
        "cell_E-Cadherin_FITC_max_mean_ratio",
    ]
    nuclei_morphology_candidates = [
        "nuclei_area",
        "nuclei_area_bbox",
        "nuclei_area_convex",
        "nuclei_area_filled",
        "nuclei_axis_major_length",
        "nuclei_axis_minor_length",
        "nuclei_equivalent_diameter_area",
        "nuclei_euler_number",
        "nuclei_extent",
        "nuclei_feret_diameter_max",
        "nuclei_solidity",
        "nuclei_inertia_tensor_eigvals-0",
        "nuclei_inertia_tensor_eigvals-1",
        "nuclei_inertia_tensor_eigvals-2",
        "nuclei_DAPI_mean_int",
        "nuclei_DAPI_min_int",
        "nuclei_DAPI_max_int",
        "nuclei_DAPI_std_int",
        "nuclei_DAPI_max_mean_ratio",
        "nuclei_DAPI_sum_int",
    ]
    organoid_morphology_candidates = [
        "organoid_area",
        "organoid_area_bbox",
        "organoid_area_convex",
        "organoid_area_filled",
        "organoid_axis_major_length",
        "organoid_axis_minor_length",
        "organoid_equivalent_diameter_area",
        "organoid_perimeter",
        "organoid_eccentricity",
        "organoid_euler_number",
        "organoid_extent",
        "organoid_feret_diameter_max",
        "organoid_solidity",
        "organoid_inertia_tensor_eigvals-0",
        "organoid_inertia_tensor_eigvals-1",
    ]
    cell_morphology_candidates = [
        "cell_area",
        "cell_area_bbox",
        "cell_area_convex",
        "cell_area_filled",
        "cell_axis_major_length",
        "cell_axis_minor_length",
        "cell_equivalent_diameter_area",
        "cell_euler_number",
        "cell_extent",
        "cell_feret_diameter_max",
        "cell_solidity",
        "cell_inertia_tensor_eigvals-0",
        "cell_inertia_tensor_eigvals-1",
        "cell_inertia_tensor_eigvals-2",
    ]
    membrane_morphology_candidates = [
        "membrane_area",
        "membrane_area_bbox",
        "membrane_area_convex",
        "membrane_area_filled",
        "membrane_axis_major_length",
        "membrane_axis_minor_length",
        "membrane_equivalent_diameter_area",
        "membrane_euler_number",
        "membrane_extent",
        "membrane_feret_diameter_max",
        "membrane_solidity",
        "membrane_inertia_tensor_eigvals-0",
        "membrane_inertia_tensor_eigvals-1",
        "membrane_inertia_tensor_eigvals-2",
    ]

    def _existing(candidates):
        return [col for col in candidates if col in df_columns]

    feature_groups = {
        "signal_intensity": _existing(signal_intensity_candidates),
        "nuclei_morphology": _existing(nuclei_morphology_candidates),
        "organoid_morphology": _existing(organoid_morphology_candidates),
        "cell_morphology": _existing(cell_morphology_candidates),
        "membrane_morphology": _existing(membrane_morphology_candidates),
    }
    jointplot_features = sorted(
        set(
            feature_groups["signal_intensity"]
            + feature_groups["nuclei_morphology"]
            + feature_groups["organoid_morphology"]
            + feature_groups["cell_morphology"]
            + feature_groups["membrane_morphology"]
        )
    )
    if not jointplot_features:
        jointplot_features = [col for col in df_columns if col not in {"well_id", "condition"}]
    return feature_groups, jointplot_features


@app.cell
def _(get_unique_values, lazy_df):
    donor_ids = get_unique_values(df=lazy_df, column_name="donor_id")
    group_ids = get_unique_values(df=lazy_df, column_name="group_number")
    return donor_ids, group_ids


@app.cell
def _(donor_ids, feature_groups, group_ids, jointplot_features, mo):
    donor_checkbox_array = mo.ui.array(
        [mo.ui.checkbox(label=donor_id, value=(i == 0)) for i, donor_id in enumerate(donor_ids)],
        label="Donor ID (multiselect)",
    )
    group_radio = mo.ui.radio(options=["None", *group_ids], value="None", label="Treatment groups")

    def _opts(options):
        return options if options else jointplot_features

    x_radio = mo.ui.radio(options=jointplot_features, value=jointplot_features[0], label="X-axis")
    y_radio = mo.ui.radio(
        options=jointplot_features,
        value=jointplot_features[1] if len(jointplot_features) > 1 else jointplot_features[0],
        label="Y-axis",
    )
    signal_feature_radio = mo.ui.radio(
        options=_opts(feature_groups["signal_intensity"]),
        value=_opts(feature_groups["signal_intensity"])[0],
        label="Signal intensity feature",
    )
    signal_plot_type_radio = mo.ui.radio(
        options=["kde", "violin"],
        value="kde",
        label="Signal intensity plot type",
    )
    nuclei_feature_radio = mo.ui.radio(
        options=_opts(feature_groups["nuclei_morphology"]),
        value=_opts(feature_groups["nuclei_morphology"])[0],
        label="Nuclei morphology feature",
    )
    organoid_feature_radio = mo.ui.radio(
        options=_opts(feature_groups["organoid_morphology"]),
        value=_opts(feature_groups["organoid_morphology"])[0],
        label="Organoid morphology feature",
    )
    cell_feature_radio = mo.ui.radio(
        options=_opts(feature_groups["cell_morphology"]),
        value=_opts(feature_groups["cell_morphology"])[0],
        label="Cell morphology feature",
    )
    membrane_feature_radio = mo.ui.radio(
        options=_opts(feature_groups["membrane_morphology"]),
        value=_opts(feature_groups["membrane_morphology"])[0],
        label="Membrane morphology feature",
    )
    aggregation_radio = mo.ui.radio(
        options=["single_cell", "organoid", "well"],
        value="well",
        label="Aggregate data (average) by:",
    )
    return (
        aggregation_radio,
        cell_feature_radio,
        donor_checkbox_array,
        group_radio,
        membrane_feature_radio,
        nuclei_feature_radio,
        organoid_feature_radio,
        signal_feature_radio,
        signal_plot_type_radio,
        x_radio,
        y_radio,
    )


@app.cell
def _(build_condition_order_by_group, lazy_df, pl):
    lazy_base = lazy_df.filter(pl.col("organoid") != 0)
    condition_order_by_group = build_condition_order_by_group(
        lazy_base.select(["group_number", "well_id", "condition"]).unique()
    )
    return condition_order_by_group, lazy_base


@app.cell
def _(OrderedDict, pl):
    _cache = OrderedDict()
    _cache_max_entries = 10
    _single_cell_sample_threshold = 120_000
    _single_cell_sample_fraction = 0.25

    def collect_plot_frame(lazy_frame, *, cache_key, columns):
        key = (cache_key, tuple(columns))
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

        frame = lazy_frame.select([pl.col(c) for c in columns]).collect()
        is_single_cell = any(isinstance(part, str) and part == "single_cell" for part in cache_key)
        if is_single_cell and frame.height > _single_cell_sample_threshold:
            frame = frame.sample(
                fraction=_single_cell_sample_fraction,
                with_replacement=False,
                shuffle=True,
                seed=42,
            )
        _cache[key] = frame
        if len(_cache) > _cache_max_entries:
            _cache.popitem(last=False)
        return frame

    return (collect_plot_frame,)


@app.cell
def _(
    aggregation_radio,
    condition_hue_order,
    condition_order_by_group,
    lazy_base,
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
        lazy_filtered = lazy_base.filter(pl.col("donor_id").cast(pl.Utf8).is_in(selected_donors))
    else:
        lazy_filtered = lazy_base

    selected_group = group_radio.value
    if selected_group != "None":
        lazy_filtered = lazy_filtered.filter(pl.col("group_number").cast(pl.Utf8) == selected_group)

    lazy_well_aggregated = (
        lazy_filtered.group_by(["donor_id", "well_id", "condition"])
        .mean()
        .drop(["organoid", "multiposition_id", "label"], strict=False)
    )
    lazy_organoid_aggregated = (
        lazy_filtered.with_columns(
            pl.concat_str(
                [pl.col("multiposition_id").cast(pl.Utf8), pl.col("organoid").cast(pl.Utf8)],
                separator="_",
            ).alias("unique_organoid_id")
        )
        .group_by(["well_id", "unique_organoid_id", "condition", "donor_id"])
        .mean()
        .drop(["organoid", "multiposition_id", "label"], strict=False)
    )
    aggregated_frames = {
        "single_cell": lazy_filtered,
        "well": lazy_well_aggregated,
        "organoid": lazy_organoid_aggregated,
    }
    lazy_plot_aggregated = aggregated_frames[aggregation_radio.value]
    present_conditions = set(
        lazy_plot_aggregated.select(pl.col("condition").cast(pl.Utf8).drop_nulls().unique())
        .collect()
        .to_series()
        .to_list()
    )
    hue_order = condition_hue_order(
        condition_order_by_group,
        selected_group=selected_group,
        present_conditions=present_conditions,
    )
    selected_donor_key = tuple(sorted(selected_donors)) if selected_donors else ("ALL",)
    state_cache_key = (selected_donor_key, selected_group, aggregation_radio.value)
    return (
        lazy_filtered,
        lazy_organoid_aggregated,
        lazy_plot_aggregated,
        hue_order,
        state_cache_key,
    )


@app.cell
def _(
    aggregation_radio,
    collect_plot_frame,
    lazy_plot_aggregated,
    donor_checkbox_array,
    group_radio,
    hue_order,
    mo,
    plt,
    signal_feature_radio,
    signal_plot_type_radio,
    sns,
    state_cache_key,
):
    _signal_df = collect_plot_frame(
        lazy_plot_aggregated,
        cache_key=(
            "signal",
            *state_cache_key,
            signal_feature_radio.value,
            signal_plot_type_radio.value,
        ),
        columns=["condition", signal_feature_radio.value],
    ).to_pandas()
    _fig, _ax = plt.subplots(figsize=(14, 8))

    if signal_plot_type_radio.value == "kde":
        sns.kdeplot(
            data=_signal_df,
            x=signal_feature_radio.value,
            hue="condition",
            fill=True,
            common_norm=False,
            alpha=0.4,
            linewidth=2,
            hue_order=hue_order or None,
            ax=_ax,
        )
        _legend = _ax.get_legend()
        if _legend is not None:
            _handles = _legend.legend_handles
            _labels = [text.get_text() for text in _legend.get_texts()]
            _legend.remove()
            _fig.legend(
                _handles,
                _labels,
                title="Treatment",
                loc="upper left",
                bbox_to_anchor=(0.66, 0.98),
                bbox_transform=_fig.transFigure,
                borderaxespad=0,
                ncol=1,
            )
        _fig.subplots_adjust(right=0.62)
    else:
        sns.violinplot(
            data=_signal_df,
            x="condition",
            y=signal_feature_radio.value,
            order=hue_order or None,
            ax=_ax,
        )
        _ax.tick_params(axis="x", rotation=25)

    _ax.set_title(f"Signal intensity ({signal_plot_type_radio.value})")
    _ax.set_xlabel(
        signal_feature_radio.value if signal_plot_type_radio.value == "kde" else "Condition"
    )
    _ax.set_ylabel(
        "Density" if signal_plot_type_radio.value == "kde" else signal_feature_radio.value
    )
    mo.vstack(
        [
            mo.md("Signal intensity KDE / violin"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    aggregation_radio,
                    group_radio,
                    signal_feature_radio,
                    signal_plot_type_radio,
                    mo.mpl.interactive(_fig),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 1, 2, 1, 12],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    aggregation_radio,
    collect_plot_frame,
    lazy_plot_aggregated,
    donor_checkbox_array,
    group_radio,
    hue_order,
    mo,
    nuclei_feature_radio,
    plt,
    sns,
    state_cache_key,
):
    _nuclei_df = collect_plot_frame(
        lazy_plot_aggregated,
        cache_key=("nuclei", *state_cache_key, nuclei_feature_radio.value),
        columns=["condition", nuclei_feature_radio.value],
    ).to_pandas()
    _fig, _ax = plt.subplots(figsize=(14, 8))
    sns.kdeplot(
        data=_nuclei_df,
        x=nuclei_feature_radio.value,
        hue="condition",
        fill=True,
        common_norm=False,
        alpha=0.4,
        linewidth=2,
        hue_order=hue_order or None,
        ax=_ax,
    )
    _legend = _ax.get_legend()
    if _legend is not None:
        _handles = _legend.legend_handles
        _labels = [text.get_text() for text in _legend.get_texts()]
        _legend.remove()
        _fig.legend(
            _handles,
            _labels,
            title="Treatment",
            loc="upper left",
            bbox_to_anchor=(0.66, 0.98),
            bbox_transform=_fig.transFigure,
            borderaxespad=0,
            ncol=1,
        )
    _ax.set_title("Nuclei morphology KDE")
    _ax.set_xlabel(nuclei_feature_radio.value)
    _ax.set_ylabel("Density")
    _fig.subplots_adjust(right=0.62)
    mo.vstack(
        [
            mo.md("Nuclei morphology KDE"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    aggregation_radio,
                    group_radio,
                    nuclei_feature_radio,
                    mo.mpl.interactive(_fig),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 1, 2, 12],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    collect_plot_frame,
    donor_checkbox_array,
    lazy_organoid_aggregated,
    group_radio,
    hue_order,
    mo,
    organoid_feature_radio,
    plt,
    sns,
    state_cache_key,
):
    _organoid_df = collect_plot_frame(
        lazy_organoid_aggregated,
        cache_key=("organoid", *state_cache_key, organoid_feature_radio.value),
        columns=["condition", organoid_feature_radio.value],
    ).to_pandas()
    _fig, _ax = plt.subplots(figsize=(14, 8))
    sns.kdeplot(
        data=_organoid_df,
        x=organoid_feature_radio.value,
        hue="condition",
        fill=True,
        common_norm=False,
        alpha=0.4,
        linewidth=2,
        hue_order=hue_order or None,
        ax=_ax,
    )
    _legend = _ax.get_legend()
    if _legend is not None:
        _handles = _legend.legend_handles
        _labels = [text.get_text() for text in _legend.get_texts()]
        _legend.remove()
        _fig.legend(
            _handles,
            _labels,
            title="Treatment",
            loc="upper left",
            bbox_to_anchor=(0.66, 0.98),
            bbox_transform=_fig.transFigure,
            borderaxespad=0,
            ncol=1,
        )
    _ax.set_title("Organoid morphology KDE")
    _ax.set_xlabel(organoid_feature_radio.value)
    _ax.set_ylabel("Density")
    _fig.subplots_adjust(right=0.62)
    mo.vstack(
        [
            mo.md("Organoid morphology KDE"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    group_radio,
                    organoid_feature_radio,
                    mo.mpl.interactive(_fig),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 2, 12],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    collect_plot_frame,
    donor_checkbox_array,
    lazy_filtered,
    group_radio,
    hue_order,
    mo,
    plt,
    cell_feature_radio,
    sns,
    state_cache_key,
):
    _cell_df = collect_plot_frame(
        lazy_filtered,
        cache_key=("cell", *state_cache_key, cell_feature_radio.value),
        columns=["condition", cell_feature_radio.value],
    ).to_pandas()
    _fig, _ax = plt.subplots(figsize=(14, 8))
    sns.kdeplot(
        data=_cell_df,
        x=cell_feature_radio.value,
        hue="condition",
        fill=True,
        common_norm=False,
        alpha=0.4,
        linewidth=2,
        hue_order=hue_order or None,
        ax=_ax,
    )
    _legend = _ax.get_legend()
    if _legend is not None:
        _handles = _legend.legend_handles
        _labels = [text.get_text() for text in _legend.get_texts()]
        _legend.remove()
        _fig.legend(
            _handles,
            _labels,
            title="Treatment",
            loc="upper left",
            bbox_to_anchor=(0.66, 0.98),
            bbox_transform=_fig.transFigure,
            borderaxespad=0,
            ncol=1,
        )
    _ax.set_title("Cell morphology KDE")
    _ax.set_xlabel(cell_feature_radio.value)
    _ax.set_ylabel("Density")
    _fig.subplots_adjust(right=0.62)
    mo.vstack(
        [
            mo.md("Cell morphology stats"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    group_radio,
                    cell_feature_radio,
                    mo.mpl.interactive(_fig),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 2, 12],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    collect_plot_frame,
    donor_checkbox_array,
    lazy_filtered,
    group_radio,
    hue_order,
    mo,
    plt,
    membrane_feature_radio,
    sns,
    state_cache_key,
):
    _membrane_df = collect_plot_frame(
        lazy_filtered,
        cache_key=("membrane", *state_cache_key, membrane_feature_radio.value),
        columns=["condition", membrane_feature_radio.value],
    ).to_pandas()
    _fig, _ax = plt.subplots(figsize=(14, 8))
    sns.kdeplot(
        data=_membrane_df,
        x=membrane_feature_radio.value,
        hue="condition",
        fill=True,
        common_norm=False,
        alpha=0.4,
        linewidth=2,
        hue_order=hue_order or None,
        ax=_ax,
    )
    _legend = _ax.get_legend()
    if _legend is not None:
        _handles = _legend.legend_handles
        _labels = [text.get_text() for text in _legend.get_texts()]
        _legend.remove()
        _fig.legend(
            _handles,
            _labels,
            title="Treatment",
            loc="upper left",
            bbox_to_anchor=(0.66, 0.98),
            bbox_transform=_fig.transFigure,
            borderaxespad=0,
            ncol=1,
        )
    _ax.set_title("Membrane morphology KDE")
    _ax.set_xlabel(membrane_feature_radio.value)
    _ax.set_ylabel("Density")
    _fig.subplots_adjust(right=0.62)
    mo.vstack(
        [
            mo.md("Membrane morphology stats"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    group_radio,
                    membrane_feature_radio,
                    mo.mpl.interactive(_fig),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 2, 12],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(donor_checkbox_array, group_radio, lazy_filtered, mo, plot_plate_view, pl):
    _counts_per_well = (
        lazy_filtered.group_by("well_id")
        .agg([pl.len().alias("total_cell_nr"), pl.max("organoid").alias("max_organoid_id")])
        .collect()
    )
    _fig_cells = plot_plate_view(
        _counts_per_well,
        column_name="total_cell_nr",
        title="Cell_number_per_well",
        label="Cell count",
        save_dir="results",
        display=False,
    )
    _fig_organoids = plot_plate_view(
        _counts_per_well,
        column_name="max_organoid_id",
        title="Organoid_number_per_well",
        label="Organoid count",
        save_dir="results",
        display=False,
    )
    mo.vstack(
        [
            mo.md("Organoid and cell numbers"),
            mo.hstack(
                [
                    donor_checkbox_array,
                    group_radio,
                    mo.vstack([mo.mpl.interactive(_fig_cells), mo.mpl.interactive(_fig_organoids)]),
                ],
                gap=3,
                align="start",
                widths=[1, 1, 8],
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
