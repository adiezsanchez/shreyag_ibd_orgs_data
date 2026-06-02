# IBD Organoid Data Analysis Pipeline

This repository contains a Pixi-managed Python pipeline for processing
organoid image-derived CSV measurements, merging them with treatment metadata,
and exploring the resulting datasets through interactive Marimo apps.

## Features

- CSV-to-parquet preprocessing for experiment folders in `raw_data/`
- Metadata merge with treatment conditions per `well_id`
- Interactive exploratory analysis with two Marimo apps:
  - `src/app.py`: correlation-oriented exploration
  - `src/app2.py`: density/morphology/treatment-focused exploration
- Polars-first data handling with selective pandas conversion for plotting
- Seaborn/Matplotlib visualization utilities for distributions and plate views
- Ruff-based linting and formatting via Pixi tasks

## Prerequisites

- [pixi](https://github.com/prefix-dev/pixi)
- Python environment resolved through Pixi (see `pyproject.toml`)

## Getting Started

1. Clone this repository:

   ```bash
   git clone <your-repo-url>
   cd shreyag_ibd_orgs_data
   ```

2. Prepare data:
   - Place experiment result folders and matching `*_conditions.csv` files in
     `raw_data/`.
   - Each run of the Marimo apps triggers merge logic that writes parquet files
     under `processed_data/` and skips already-generated outputs.

3. Start an analysis app:

   ```bash
   pixi run correlation_plots
   ```

   or

   ```bash
   pixi run density_plots
   ```

## Pipeline Overview

1. **Ingest**: Read per-experiment CSV files from `raw_data/<experiment_id>/`.
2. **Merge**: Join measurements with condition metadata from
   `raw_data/<experiment_id>_conditions.csv`.
3. **Persist**: Save merged datasets as parquet files in `processed_data/`.
4. **Explore**: Use Marimo UIs to filter by donor/treatment group, aggregate
   data (`single_cell`, `organoid`, `well`), and render interactive plots.

## Development

### Launch editor

```bash
pixi run edit
```

### Run tests

```bash
pixi run test
```

### Lint and format

```bash
pixi run lint
pixi run format
```

### Install pre-commit hooks

```bash
pixi run pre-commit-install
```

## Project Structure

```text
├── raw_data/                 # Input experiment folders and condition CSVs
├── processed_data/           # Generated merged parquet files
├── src/
│   ├── app.py                # Correlation-focused Marimo app
│   ├── app2.py               # Density/morphology Marimo app
│   ├── utils_data_analysis.py
│   └── utils_data_plotting.py
├── pyproject.toml            # Pixi tasks, dependencies, tool configuration
└── pixi.lock                 # Locked dependency resolution
```

## License

This project is licensed under **BSL-3**.