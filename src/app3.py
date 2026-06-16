import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import warnings
    from pathlib import Path

    warnings.filterwarnings(
        "ignore",
        message="Found Intel OpenMP \\('libiomp'\\) and LLVM OpenMP \\('libomp'\\) loaded at.*",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="n_jobs value 1 overridden to 1 by setting random_state.*",
        category=UserWarning,
    )

    import igraph as ig
    import leidenalg
    import marimo as mo
    import plotly.express as px
    import polars as pl
    from scipy import sparse
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    from utils_data_analysis import (
        build_condition_order_by_group,
        condition_hue_order,
        get_unique_values,
        merge_csv_files,
    )

    return (
        AgglomerativeClustering,
        KMeans,
        PCA,
        Path,
        StandardScaler,
        TSNE,
        UMAP,
        build_condition_order_by_group,
        condition_hue_order,
        get_unique_values,
        ig,
        leidenalg,
        merge_csv_files,
        mo,
        pl,
        px,
        sparse,
        kneighbors_graph,
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
def _(lazy_df, pl):
    metadata_columns = {
        "well_id",
        "condition",
        "group_number",
        "donor_id",
        "organoid",
        "multiposition_id",
        "label",
    }
    schema = lazy_df.collect_schema()
    numeric_columns = [
        col
        for col, dtype in schema.items()
        if col not in metadata_columns and dtype.is_numeric() and dtype != pl.Boolean
    ]
    return (numeric_columns,)


@app.cell
def _(get_unique_values, lazy_df):
    donor_ids = get_unique_values(df=lazy_df, column_name="donor_id")
    group_ids = get_unique_values(df=lazy_df, column_name="group_number")
    return donor_ids, group_ids


@app.cell
def _(donor_ids, group_ids, mo):
    donor_checkbox_array = mo.ui.array(
        [mo.ui.checkbox(label=donor_id, value=(i == 0)) for i, donor_id in enumerate(donor_ids)],
        label="Donor ID (multiselect)",
    )
    group_radio = mo.ui.radio(options=["None", *group_ids], value="None", label="Treatment groups")
    aggregation_radio = mo.ui.radio(
        options=["single_cell", "organoid", "well"],
        value="well",
        label="Aggregate data (average) by:",
    )
    return aggregation_radio, donor_checkbox_array, group_radio


@app.cell
def _(mo, numeric_columns):
    max_features = min(50, max(len(numeric_columns), 1))
    default_features = min(12, max_features)

    n_features_slider = mo.ui.slider(
        start=1,
        stop=max_features,
        step=1,
        value=default_features,
        label="Number of numeric features used",
    )
    sample_size_slider = mo.ui.slider(
        start=500,
        stop=40000,
        step=500,
        value=8000,
        label="Maximum points for embedding/clustering",
    )
    random_seed_number = mo.ui.number(start=0, stop=999999, value=42, label="Random seed")
    kmeans_k_slider = mo.ui.slider(start=2, stop=12, step=1, value=4, label="k-means clusters")
    hierarchical_k_slider = mo.ui.slider(
        start=2, stop=12, step=1, value=4, label="Hierarchical clusters"
    )
    leiden_k_neighbors_slider = mo.ui.slider(
        start=5, stop=50, step=1, value=15, label="Leiden graph neighbors"
    )
    tsne_perplexity_slider = mo.ui.slider(
        start=5, stop=80, step=1, value=30, label="t-SNE perplexity"
    )
    umap_n_neighbors_slider = mo.ui.slider(
        start=5, stop=100, step=1, value=20, label="UMAP neighbors"
    )
    return (
        hierarchical_k_slider,
        kmeans_k_slider,
        leiden_k_neighbors_slider,
        n_features_slider,
        random_seed_number,
        sample_size_slider,
        tsne_perplexity_slider,
        umap_n_neighbors_slider,
    )


@app.cell
def _(n_features_slider, numeric_columns):
    feature_count = min(n_features_slider.value, len(numeric_columns))
    selected_numeric_columns = numeric_columns[:feature_count]
    return feature_count, selected_numeric_columns


@app.cell
def _(
    aggregation_radio,
    donor_checkbox_array,
    feature_count,
    group_radio,
    hierarchical_k_slider,
    kmeans_k_slider,
    leiden_k_neighbors_slider,
    mo,
    n_features_slider,
    random_seed_number,
    sample_size_slider,
    selected_numeric_columns,
    tsne_perplexity_slider,
    umap_n_neighbors_slider,
):
    if feature_count > 10:
        extra = feature_count - 10
        feature_preview = (
            ", ".join(selected_numeric_columns[:10]) + f", … (+{extra} more)"
        )
    else:
        feature_preview = ", ".join(selected_numeric_columns)

    mo.vstack(
        [
            mo.md("# Dimensionality reduction and clustering explorer"),
            mo.md(
                """
                **Algorithm quick guide**
                - **PCA (linear DR):** projects features onto orthogonal directions
                  that explain the most variance.
                - **t-SNE (non-linear DR):** preserves local neighborhoods to reveal
                  fine-grained manifolds.
                - **UMAP (non-linear DR):** preserves local structure while retaining
                  more global geometry than t-SNE.
                - **k-means (partition clustering):** assigns points to `k`
                  centroid-based clusters.
                - **Hierarchical clustering (agglomerative):** merges nearest groups
                  iteratively into a cluster tree.
                - **Leiden (graph clustering):** detects communities on a kNN graph
                  for topology-aware clusters.
                """
            ),
            mo.md(
                "Hover points to inspect donor and treatment "
                "alongside cluster memberships."
            ),
            mo.md(
                f"**Features in this run ({feature_count}):** {feature_preview}"
            ),
            mo.md(
                """
                **Data selection controls**
                - **Donor ID:** choose which patient-derived samples to include.
                - **Treatment groups:** focus on one experimental group, or `None`
                  to keep all groups.
                - **Aggregate data by:** average measurements at the well, organoid,
                  or single-cell level before analysis.
                - **Number of numeric features:** how many measured image features
                  (e.g. membrane area, shape metrics) feed into the analysis.
                  More features capture more biology but can add noise.
                - **Maximum points:** cap how many data points are plotted and
                  clustered. Lower values run faster; higher values use more of
                  your dataset.
                - **Random seed:** fixes random choices (sampling and algorithms)
                  so results are reproducible when you rerun with the same settings.

                **Clustering controls**
                - **k-means clusters:** how many groups k-means should split the
                  data into.
                - **Hierarchical clusters:** how many final groups to cut from the
                  hierarchical tree.
                - **Leiden graph neighbors:** how many nearest neighbors each point
                  connects to when building the graph for Leiden clustering.
                  Higher values merge broader regions.

                **Embedding controls**
                - **t-SNE perplexity:** balances local vs. global structure in the
                  t-SNE map. Try 5–50; very small datasets use a lower effective
                  value automatically.
                - **UMAP neighbors:** how locally vs. globally UMAP shapes the map.
                  Lower values emphasize fine structure; higher values emphasize
                  broader trends.
                """
            ),
            mo.hstack(
                [
                    donor_checkbox_array,
                    group_radio,
                    aggregation_radio,
                    n_features_slider,
                    sample_size_slider,
                    random_seed_number,
                ],
                gap=2,
                widths=[1, 1, 1, 1, 1, 1],
            ),
            mo.hstack(
                [
                    kmeans_k_slider,
                    hierarchical_k_slider,
                    leiden_k_neighbors_slider,
                    tsne_perplexity_slider,
                    umap_n_neighbors_slider,
                ],
                gap=2,
                widths=[1, 1, 1, 1, 1],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(build_condition_order_by_group, lazy_df, pl):
    lazy_base = lazy_df.filter(pl.col("organoid") != 0)
    condition_order_by_group = build_condition_order_by_group(
        lazy_base.select(["group_number", "well_id", "condition"]).unique()
    )
    return condition_order_by_group, lazy_base


@app.cell
def _(
    aggregation_radio,
    condition_hue_order,
    condition_order_by_group,
    donor_checkbox_array,
    donor_ids,
    group_radio,
    lazy_base,
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
    return hue_order, lazy_plot_aggregated, selected_group


@app.cell
def _(
    StandardScaler,
    TSNE,
    UMAP,
    AgglomerativeClustering,
    KMeans,
    PCA,
    hierarchical_k_slider,
    ig,
    kmeans_k_slider,
    kneighbors_graph,
    leiden_k_neighbors_slider,
    leidenalg,
    pl,
    random_seed_number,
    sample_size_slider,
    selected_numeric_columns,
    sparse,
    tsne_perplexity_slider,
    umap_n_neighbors_slider,
    lazy_plot_aggregated,
):
    empty_plot_df = pl.DataFrame(
        {
            "donor_id": [],
            "condition": [],
            "pca_1": [],
            "pca_2": [],
            "tsne_1": [],
            "tsne_2": [],
            "umap_1": [],
            "umap_2": [],
            "cluster_kmeans": [],
            "cluster_hierarchical": [],
            "cluster_leiden": [],
        }
    )

    if not selected_numeric_columns:
        plot_df = empty_plot_df
        method_note = "No numeric feature columns found."
    else:
        required_columns = ["donor_id", "condition", *selected_numeric_columns]
        collected = lazy_plot_aggregated.select(required_columns).drop_nulls().collect()

        if collected.height == 0:
            plot_df = empty_plot_df
            method_note = "No rows available after filtering and dropping null feature values."
        else:
            max_points = sample_size_slider.value
            if collected.height > max_points:
                collected = collected.sample(
                    n=max_points,
                    shuffle=True,
                    seed=random_seed_number.value,
                )

            pdf = collected.to_pandas()
            X = pdf[selected_numeric_columns].to_numpy()
            X_scaled = StandardScaler().fit_transform(X)

            pca_model = PCA(n_components=2, random_state=random_seed_number.value)
            pca_coords = pca_model.fit_transform(X_scaled)

            tsne_perplexity = min(tsne_perplexity_slider.value, max(5, len(pdf) // 3))
            tsne_model = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                random_state=random_seed_number.value,
                perplexity=tsne_perplexity,
            )
            tsne_coords = tsne_model.fit_transform(X_scaled)

            umap_model = UMAP(
                n_components=2,
                random_state=random_seed_number.value,
                n_neighbors=min(umap_n_neighbors_slider.value, max(2, len(pdf) - 1)),
                min_dist=0.1,
            )
            umap_coords = umap_model.fit_transform(X_scaled)

            kmeans_labels = (
                KMeans(
                    n_clusters=kmeans_k_slider.value,
                    random_state=random_seed_number.value,
                )
                .fit(X_scaled)
                .labels_
            )
            hierarchical_labels = AgglomerativeClustering(
                n_clusters=hierarchical_k_slider.value,
                linkage="ward",
            ).fit(X_scaled).labels_

            n_neighbors = min(leiden_k_neighbors_slider.value, max(2, len(pdf) - 1))
            knn = kneighbors_graph(X_scaled, n_neighbors=n_neighbors, mode="connectivity")
            adjacency = sparse.triu(knn, k=1).tocoo()
            graph = ig.Graph(
                n=len(pdf),
                edges=list(zip(adjacency.row.tolist(), adjacency.col.tolist(), strict=False)),
                directed=False,
            )
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                seed=random_seed_number.value,
            )
            leiden_labels = partition.membership

            plot_df = pl.DataFrame(
                {
                    "donor_id": pdf["donor_id"].astype(str).to_list(),
                    "condition": pdf["condition"].astype(str).to_list(),
                    "pca_1": pca_coords[:, 0].tolist(),
                    "pca_2": pca_coords[:, 1].tolist(),
                    "tsne_1": tsne_coords[:, 0].tolist(),
                    "tsne_2": tsne_coords[:, 1].tolist(),
                    "umap_1": umap_coords[:, 0].tolist(),
                    "umap_2": umap_coords[:, 1].tolist(),
                    "cluster_kmeans": [f"KM_{c}" for c in kmeans_labels],
                    "cluster_hierarchical": [f"HC_{c}" for c in hierarchical_labels],
                    "cluster_leiden": [f"LD_{c}" for c in leiden_labels],
                }
            )
            method_note = (
                f"Rows embedded: {plot_df.height}. Features used: {len(selected_numeric_columns)}."
                f" t-SNE perplexity: {tsne_perplexity}. "
                f"Leiden neighbors: {n_neighbors}."
            )
    return plot_df, selected_numeric_columns, method_note


@app.cell
def _(hue_order, method_note, mo, plot_df, px):
    _plot_pdf = plot_df.to_pandas()
    _base_hover = [
        "donor_id",
        "condition",
        "cluster_kmeans",
        "cluster_hierarchical",
        "cluster_leiden",
    ]

    pca_treatment_fig = px.scatter(
        _plot_pdf,
        x="pca_1",
        y="pca_2",
        color="condition",
        category_orders={"condition": hue_order} if hue_order else None,
        hover_data=_base_hover,
        title="PCA (linear DR) colored by treatment",
    )
    pca_treatment_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    pca_kmeans_fig = px.scatter(
        _plot_pdf,
        x="pca_1",
        y="pca_2",
        color="cluster_kmeans",
        hover_data=_base_hover,
        title="PCA (linear DR) with k-means clusters",
    )
    pca_kmeans_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    pca_hierarchical_fig = px.scatter(
        _plot_pdf,
        x="pca_1",
        y="pca_2",
        color="cluster_hierarchical",
        hover_data=_base_hover,
        title="PCA (linear DR) with hierarchical clusters",
    )
    pca_hierarchical_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    pca_leiden_fig = px.scatter(
        _plot_pdf,
        x="pca_1",
        y="pca_2",
        color="cluster_leiden",
        hover_data=_base_hover,
        title="PCA (linear DR) with Leiden clusters",
    )
    pca_leiden_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    mo.vstack(
        [
            mo.md("## Linear concepts: PCA + clustering"),
            mo.md(method_note),
            mo.hstack(
                [
                    mo.ui.plotly(pca_treatment_fig),
                    mo.ui.plotly(pca_kmeans_fig),
                ],
                widths=[1, 1],
                gap=2,
            ),
            mo.hstack(
                [
                    mo.ui.plotly(pca_hierarchical_fig),
                    mo.ui.plotly(pca_leiden_fig),
                ],
                widths=[1, 1],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(hue_order, mo, plot_df, px):
    _plot_pdf = plot_df.to_pandas()
    _base_hover = [
        "donor_id",
        "condition",
        "cluster_kmeans",
        "cluster_hierarchical",
        "cluster_leiden",
    ]

    tsne_treatment_fig = px.scatter(
        _plot_pdf,
        x="tsne_1",
        y="tsne_2",
        color="condition",
        category_orders={"condition": hue_order} if hue_order else None,
        hover_data=_base_hover,
        title="t-SNE (non-linear DR) colored by treatment",
    )
    tsne_treatment_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    tsne_kmeans_fig = px.scatter(
        _plot_pdf,
        x="tsne_1",
        y="tsne_2",
        color="cluster_kmeans",
        hover_data=_base_hover,
        title="t-SNE with k-means clusters",
    )
    tsne_kmeans_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    umap_treatment_fig = px.scatter(
        _plot_pdf,
        x="umap_1",
        y="umap_2",
        color="condition",
        category_orders={"condition": hue_order} if hue_order else None,
        hover_data=_base_hover,
        title="UMAP (non-linear DR) colored by treatment",
    )
    umap_treatment_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    umap_leiden_fig = px.scatter(
        _plot_pdf,
        x="umap_1",
        y="umap_2",
        color="cluster_leiden",
        hover_data=_base_hover,
        title="UMAP with Leiden clusters",
    )
    umap_leiden_fig.update_traces(marker={"size": 6, "opacity": 0.65})

    mo.vstack(
        [
            mo.md("## Non-linear concepts: t-SNE and UMAP"),
            mo.hstack(
                [
                    mo.ui.plotly(tsne_treatment_fig),
                    mo.ui.plotly(tsne_kmeans_fig),
                ],
                widths=[1, 1],
                gap=2,
            ),
            mo.hstack(
                [
                    mo.ui.plotly(umap_treatment_fig),
                    mo.ui.plotly(umap_leiden_fig),
                ],
                widths=[1, 1],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
