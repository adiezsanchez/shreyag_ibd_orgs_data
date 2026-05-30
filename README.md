---
title: Readme
marimo-version: 0.23.8
---

# marimo + pixi Starter Template

A starter template for [marimo](https://marimo.io) notebooks using [pixi](https://github.com/prefix-dev/pixi) for dependency and project management. This template provides a modern Python development setup with best practices for notebook development.

## Features

- 🚀 Python 3.12+ support
- 📦 Fast dependency management with `pixi`
- 🧪 Testing setup with pytest
- 🎯 Code quality with Ruff (linting + formatting)
- 👷 CI/CD with GitHub Actions
- 📓 Interactive notebook development with marimo

## Prerequisites

- [pixi](https://github.com/prefix-dev/pixi) installed

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/marimo-pixi-starter-template
   cd marimo-pixi-starter-template
   ```

2. Run the marimo editor:

   ```bash
   pixi run edit
   ```

## Development

### Running Tests

```bash
pixi run test
```

### Linting and formatting

```bash
pixi run lint
pixi run format
```

### Install pre-commit

```bash
pixi run pre-commit-install
```

## Project Structure

```markdown
├── .github/          # GitHub Actions workflows
├── src/              # Source code
│   └── app.py        # Sample marimo notebook
├── tests/            # Test files
├── pyproject.toml    # Project configuration
└── pixi.lock         # Dependency lock file
```

## License

MIT