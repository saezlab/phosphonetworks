"""Utility helpers for handling data, similarity metrics, and network prep."""

from __future__ import annotations

import itertools
import os
import tempfile
import zipfile
from typing import Dict, Iterable, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import requests

from phosphonetworks import config


import os
import zipfile
import tempfile
from typing import Optional

import requests
from tqdm import tqdm 


def download_manuscript_data(target_dir: str = "data", chunk_size: int = 1024 * 128) -> None:
    """Download the publicly hosted dataset and extract it into ``target_dir`` with progress bars."""

    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists. Skipping download.")
        return True
    
    print('Downloading manuscript data into', target_dir)

    url = "https://zenodo.org/records/17161034/files/phosphonetworks_data.zip?download=1"
    print(f"Downloading data from {url}...")

    tmp_path: Optional[str] = None
    try:
        # --- Download with progress -----------------------------------------------------
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))  # 0 if unknown
            # NamedTemporaryFile(delete=False) so we can reopen after closing context
            with tempfile.NamedTemporaryFile(prefix="scape_", suffix=".zip", delete=False) as tmp, \
                 tqdm(
                     total=total if total > 0 else None,
                     unit="B",
                     unit_scale=True,
                     unit_divisor=1024,
                     desc="Downloading",
                     leave=False,
                 ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    tmp.write(chunk)
                    pbar.update(len(chunk))
                tmp_path = tmp.name

        print(f"Download complete: {tmp_path}")

        # --- Prepare target dir ---------------------------------------------------------
        os.makedirs(target_dir, exist_ok=False)

        def _is_within_directory(directory: str, target: str) -> bool:
            abs_dir = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_dir]) == os.path.commonpath([abs_dir, abs_target])

        # --- Extract with progress ------------------------------------------------------
        with zipfile.ZipFile(tmp_path) as archive:
            members = archive.infolist()

            # Pre-check: refuse path traversal
            for member in members:
                dest_path = os.path.join(target_dir, member.filename)
                if not _is_within_directory(target_dir, dest_path):
                    raise RuntimeError(
                        "Refusing to extract outside target dir: %s" % member.filename
                    )

            # Show progress by uncompressed size if available; otherwise by file count
            total_bytes = sum(m.file_size for m in members)
            by_size = total_bytes > 0

            with tqdm(
                total=(total_bytes if by_size else len(members)),
                unit=("B" if by_size else "file"),
                unit_scale=by_size,
                unit_divisor=1024,
                desc="Extracting",
                leave=False,
            ) as pbar:
                for m in members:
                    # Extract each member manually so we can update the bar incrementally
                    source = archive.open(m)
                    dest_path = os.path.join(target_dir, m.filename)

                    if m.is_dir():
                        os.makedirs(dest_path, exist_ok=True)
                        pbar.update(m.file_size if by_size else 1)
                        continue

                    # Ensure parent dir exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    with open(dest_path, "wb") as f:
                        # Stream the file out in chunks while updating the bar
                        remaining = m.file_size
                        while True:
                            buf = source.read(min(chunk_size, remaining if remaining else chunk_size))
                            if not buf:
                                break
                            f.write(buf)
                            if by_size:
                                pbar.update(len(buf))
                        if not by_size:
                            pbar.update(1)

                    source.close()

        print(f"Extraction complete: {target_dir}")

    except requests.RequestException as exc:
        print(f"Download failed: {exc}")
        if os.path.isdir(target_dir) and not os.listdir(target_dir):
            try:
                os.rmdir(target_dir)
            except OSError:
                pass
        raise
    except zipfile.BadZipFile as exc:
        print(f"ZIP error: {exc}")
        if os.path.isdir(target_dir) and not os.listdir(target_dir):
            try:
                os.rmdir(target_dir)
            except OSError:
                pass
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def get_uniprot_symbol_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return bidirectional UniProt ↔ gene symbol lookup dictionaries."""

    translator_file = os.path.join(config.DATA_DIR, "translator_df.csv")
    translator_df = pd.read_csv(translator_file)
    translator_df = translator_df.dropna(subset=["symbol", "uniprot"])
    uniprot_to_symbol = translator_df.set_index("uniprot")["symbol"].to_dict()
    symbol_to_uniprot = translator_df.set_index("symbol")["uniprot"].to_dict()
    return uniprot_to_symbol, symbol_to_uniprot

def pairwise_matrix_to_pairwise_df(
    pairwise_matrix: pd.DataFrame,
    var_name: str = "kinase",
    value_name: str = "score",
) -> pd.DataFrame:
    """Convert a symmetric pairwise matrix into a long-form DataFrame."""

    pairwise_matrix = pairwise_matrix.rename_axis(None, axis=0).rename_axis(None, axis=1)
    mask = np.triu(np.ones(pairwise_matrix.shape, dtype=bool), k=1)
    filtered = pairwise_matrix.where(mask)

    pairwise_df = (
        filtered.stack()
        .reset_index()
        .rename(columns={"level_0": f"{var_name}_a", "level_1": f"{var_name}_b", 0: value_name})
    )
    return pairwise_df

def flex_site_mapping(input_list: Iterable[str]) -> list[str]:
    """Map between UniProt IDs and symbols while retaining site suffixes."""

    items = list(input_list)
    prots = [item.split("_")[0] for item in items]
    sites = [item.split("_")[1] if "_" in item else None for item in items]

    uniprot_to_symbol, symbol_to_uniprot = get_uniprot_symbol_mapping()

    uniprot_hits = sum(1 for item in prots[:10] if item in uniprot_to_symbol)
    symbol_hits = sum(1 for item in prots[:10] if item in symbol_to_uniprot)
    mapping = uniprot_to_symbol if uniprot_hits > symbol_hits else symbol_to_uniprot

    mapped_prots = [mapping.get(item, item) for item in prots]
    mapped_sites = [f"{prot}_{site}" if site else prot for prot, site in zip(mapped_prots, sites)]
    return mapped_sites

def overlap_from_df(df: pd.DataFrame, metric: str = "jaccard") -> pd.DataFrame:
    """Compute pairwise identification overlap between study columns.

    Args:
        df: Wide table with studies as columns and sites as rows.
        metric: Either ``"jaccard"`` or ``"overlap"`` (Szymkiewicz–Simpson).

    Returns:
        Long-form DataFrame with one row per study pair and overlap statistics.
    """
    metric = metric.lower()
    if metric not in {"jaccard", "overlap"}:
        raise ValueError("metric must be 'jaccard' or 'overlap'")

    cols = list(df.columns)
    id_masks = {c: df[c].notna() for c in cols}  # boolean identification masks

    rows = []
    for a, b in itertools.combinations(cols, 2):
        mask_a = id_masks[a]
        mask_b = id_masks[b]
        n_a = int(mask_a.sum())
        n_b = int(mask_b.sum())
        n_shared = int((mask_a & mask_b).sum())
        n_union = int((mask_a | mask_b).sum())

        if metric == "jaccard":
            score = np.nan if n_union == 0 else n_shared / n_union
        else:  # overlap
            denom = min(n_a, n_b)
            score = np.nan if denom == 0 else n_shared / denom

        rows.append({
            "study_a": a,
            "study_b": b,
            "score": score,
            "n_a": n_a,
            "n_b": n_b,
            "n_shared": n_shared,
            "n_union": n_union
        })

    return pd.DataFrame(rows)


def corr_from_df(
    df: pd.DataFrame, min_pairs: int = 10, method: str = "spearman"
) -> pd.DataFrame:
    """Compute pairwise correlations between study columns.

    Args:
        df: Wide table with studies as columns and sites as rows.
        min_pairs: Minimum number of shared quantifications required.
        method: Correlation method accepted by :meth:`pandas.Series.corr`.

    Returns:
        Long-form DataFrame with correlations and supporting counts.
    """
    cols = list(df.columns)
    col_name_map = {"spearman": "spearman_rho", "pearson": "pearson_r"}
    corr_column = col_name_map.get(method.lower(), f"{method.lower()}_corr")

    rows = []
    for a, b in itertools.combinations(cols, 2):
        s1 = df[a]
        s2 = df[b]
        mask = s1.notna() & s2.notna()
        n_pairs = int(mask.sum())

        if n_pairs >= min_pairs:
            # Pandas' spearman uses pairwise complete obs on the aligned Series
            rho = s1[mask].corr(s2[mask], method=method)
        else:
            rho = np.nan

        rows.append({
            "study_a": a,
            "study_b": b,
            corr_column: rho,
            "n_pairs": n_pairs,
            "method": method.lower(),
        })
    outdf = pd.DataFrame(rows)
    return outdf

def kinsub_to_pkn(
    kinsub: pd.DataFrame,
    kinases: Iterable[str],
    int_resource: Optional[str] = None,
) -> nx.DiGraph:
    """Convert a kinase-substrate table to a kinase-kinase PKN graph."""

    if int_resource is not None:
        pkn = kinsub[kinsub["resource"] == int_resource].copy()
    else:
        pkn = kinsub.copy()

    pkn["target"] = pkn["target"].str.replace("_.*", "", regex=True)
    pkn = pkn[pkn["source"].isin(kinases) & pkn["target"].isin(kinases)]
    pkn = pkn[pkn["source"] != pkn["target"]].drop_duplicates()
    pkn = (
        pkn.groupby(["source", "target"])["score"].mean().reset_index().rename(columns={"score": "weight"})
    )

    weight_min = float(pkn["weight"].min())
    weight_max = float(pkn["weight"].max())
    if weight_max > weight_min:
        pkn["weight"] = (pkn["weight"] - weight_min) / (weight_max - weight_min)
    else:
        pkn["weight"] = 1.0

    return nx.from_pandas_edgelist(
        pkn, "source", "target", ["weight"], create_using=nx.DiGraph()
    )

def kinact_to_terminals(
    kinact: pd.DataFrame, int_study: Optional[str] = None
) -> Dict[str, float]:
    """Extract absolute kinase activities ready for network algorithms."""

    if int_study is not None:
        terminals = kinact[kinact["study"] == int_study].copy()
    else:
        terminals = kinact.copy()

    terminals = terminals[["kinase", "activity"]]
    terminals["abs_activity"] = terminals["activity"].abs()
    return terminals.set_index("kinase")["abs_activity"].to_dict()

def prepare_pkn_and_terminals(
    pkn: nx.DiGraph,
    terminals: Mapping[str, float],
    root: str = "P00533",
    verbose: bool = True,
) -> tuple[nx.DiGraph, Dict[str, float]]:
    """Filter a PKN and terminals to match reachable kinases from the root."""

    if verbose:
        print(f"Initial number of terminals: {len(terminals)}")

    filt_terminals = {k: v for k, v in terminals.items() if k in pkn.nodes()}
    if verbose:
        print(f"[FILTER TO NETWORK NODES]: {len(filt_terminals)}")

    reachable = nx.descendants(pkn, root)
    reachable.add(root)
    filt_terminals = {k: v for k, v in filt_terminals.items() if k in reachable}
    if verbose:
        print(f"[FILTER TO REACHABLE FROM ROOT {root}]: {len(filt_terminals)}")
        print(f"Final number of terminals: {len(filt_terminals)}")

    filt_pkn = pkn.subgraph(reachable).copy()
    if verbose:
        print(
            f"PKN has {filt_pkn.number_of_nodes()} nodes and {filt_pkn.number_of_edges()} edges"
        )

    if filt_terminals:
        min_val = float(min(filt_terminals.values()))
        max_val = float(max(filt_terminals.values()))
        if verbose:
            print(f"Min terminal value: {min_val}")
            print(f"Max terminal value: {max_val}")
        if max_val > min_val:
            filt_terminals = {
                k: (v - min_val) / (max_val - min_val) for k, v in filt_terminals.items()
            }
        else:
            filt_terminals = {k: 1.0 for k in filt_terminals}

    return filt_pkn, dict(filt_terminals)
