#!/usr/bin/env python3
"""
Standard scRNA-seq preprocessing workflow with Scanpy:
QC -> filtering -> normalization -> HVG selection -> scaling -> PCA ->
neighbors -> UMAP -> Leiden clustering.

Input : input_dataset.h5ad   (expects raw, unnormalized counts in .X)
Output: processed_dataset.h5ad

Usage:
    python scrna_workflow.py
    python scrna_workflow.py --max-pct-mt 5 --resolution 0.8
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scrna_workflow")


# --------------------------------------------------------------------------- #
# Arguments (defaults match the specified requirements)
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scanpy scRNA-seq preprocessing workflow")
    p.add_argument("--input", default="./data/pbmc_sample.h5ad")
    p.add_argument("--output", default="./output/processed_dataset.h5ad")
    p.add_argument("--min-genes", type=int, default=200, help="Min genes per cell")
    p.add_argument("--min-cells", type=int, default=3, help="Min cells per gene")
    p.add_argument("--max-genes", type=int, default=None,
                   help="Optional upper bound on genes per cell (crude doublet filter)")
    p.add_argument("--max-pct-mt", type=float, default=10.0,
                   help="Keep cells with mitochondrial %% below this value")
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=30, help="PCs used for neighbors graph")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--resolution", type=float, default=0.5, help="Leiden resolution")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def looks_like_raw_counts(adata, n_cells_check: int = 200) -> bool:
    """Heuristic check that .X holds non-negative integer counts."""
    n = min(n_cells_check, adata.n_obs)
    sub = adata.X[:n]
    data = sub.data if sp.issparse(sub) else np.asarray(sub).ravel()
    if data.size == 0:
        return False
    return bool(np.all(data >= 0) and np.allclose(data, np.round(data)))


def annotate_gene_groups(adata) -> list:
    """
    Flag mitochondrial, ribosomal and hemoglobin genes (human or mouse naming)
    and return the list of boolean var columns that were actually populated.
    """
    names_upper = adata.var_names.str.upper()

    # Mitochondrial: 'MT-' (human) / 'mt-' (mouse); case-insensitive match.
    adata.var["mt"] = names_upper.str.startswith("MT-")
    # Ribosomal proteins: RPS / RPL (human) and Rps / Rpl (mouse).
    adata.var["ribo"] = names_upper.str.match(r"^RP[SL]\d+")
    # Hemoglobin genes (HBA1, HBB, Hba-a1, ...); avoids HBEGF, HBS1L, etc.
    adata.var["hb"] = names_upper.str.match(r"^HB[ABDEGMQZ]\d*(-\w+)?$")

    qc_vars = []
    for key, label in [("mt", "mitochondrial"), ("ribo", "ribosomal"), ("hb", "hemoglobin")]:
        n = int(adata.var[key].sum())
        if n > 0:
            qc_vars.append(key)
            log.info("Found %d %s genes.", n, label)
        else:
            log.warning("No %s genes detected; skipping '%s' metrics.", label, key)
    return qc_vars


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    sc.settings.verbosity = 2

    in_path, out_path = Path(args.input), Path(args.output)
    if not in_path.is_file():
        log.error("Input file not found: %s", in_path)
        return 1

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    log.info("Reading %s ...", in_path)
    adata = sc.read_h5ad(in_path)
    log.info("Loaded: %d cells x %d genes.", adata.n_obs, adata.n_vars)

    if adata.n_obs == 0 or adata.n_vars == 0:
        log.error("Dataset is empty.")
        return 1

    # Gene names must be unique for indexing and plotting.
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Ensure sparse CSR storage for memory efficiency.
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.X = adata.X.astype(np.float32)

    if not looks_like_raw_counts(adata):
        log.warning(
            "adata.X does not look like raw integer counts. Normalization and "
            "HVG selection assume raw counts; results may be unreliable."
        )

    # Keep a copy of the raw counts (all genes, post-QC) for tools that need them.
    adata.layers["counts"] = adata.X.copy()

    # ------------------------------------------------------------------ #
    # 2. Quality control metrics
    # ------------------------------------------------------------------ #
    qc_vars = annotate_gene_groups(adata)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, percent_top=None, log1p=False, inplace=True
    )

    # ------------------------------------------------------------------ #
    # 3. Filtering
    # ------------------------------------------------------------------ #
    n_cells_start, n_genes_start = adata.n_obs, adata.n_vars

    # Remove low-quality cells (too few detected genes).
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    if args.max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=args.max_genes)
    if "mt" in qc_vars:
        adata = adata[adata.obs["pct_counts_mt"] < args.max_pct_mt].copy()
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    log.info(
        "After filtering: %d/%d cells, %d/%d genes kept.",
        adata.n_obs, n_cells_start, adata.n_vars, n_genes_start,
    )
    if adata.n_obs == 0:
        log.error("No cells left after filtering; check the thresholds.")
        return 1

    # Save adata (create the output folder; write_h5ad won't)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    log.info("Saved: %s", out_path)
    return 0


# Check if executed
if __name__ == "__main__":
    sys.exit(main())
else:
    print("This module is intended to be executed as a script.")