"""Integration test mirroring the paper reproduction notebook."""

from __future__ import annotations

import os
from pathlib import Path

import phosphonetworks as pp


def _prepare_data_dir() -> Path:
    """Ensure manuscript data are downloaded and return the cache directory."""

    base_dir = Path(
        os.environ.get("PHOSPHONETWORKS_DATA_DIR", "~/.cache/phosphonetworks_dataset")
    ).expanduser()
    pp.utils.download_manuscript_data(str(base_dir))
    cache_root = base_dir / "phosphonetworks_data"
    if not cache_root.exists():
        raise FileNotFoundError(f"Expected cache directory at {cache_root} after download")
    pp.config.CACHE_DIR = os.path.join(str(cache_root), "")
    figures_dir = Path(os.environ.get("PHOSPHONETWORKS_FIGURES_DIR", "figures"))
    pp.config.FIGURES_DIR = str(figures_dir)
    return cache_root


def test_paper_pipeline_runs(tmp_path: Path) -> None:
    """Execute the sequential paper pipelines end-to-end."""

    # Route figures to a throwaway location to avoid polluting the workspace in CI.
    figures_dir = tmp_path / "figures"
    os.environ["PHOSPHONETWORKS_FIGURES_DIR"] = str(figures_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")

    _prepare_data_dir()

    pipelines = [
        pp.pipelines.run_kinsub_pipeline,
        pp.pipelines.run_site_pipeline,
        pp.pipelines.run_egf_kin_pipeline,
        pp.pipelines.run_egf_gt_pipeline,
        pp.pipelines.run_net_benchmark_pipeline,
    ]

    for pipeline in pipelines:
        pipeline()
