"""Utilities for a compact CYP inhibition uncertainty study.

These functions intentionally avoid using the challenge's blinded test labels.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score


ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def find_smiles_column(df: pd.DataFrame) -> str:
    """Find a likely SMILES column without assuming exact capitalization."""
    preferred = ("SMILES", "smiles", "canonical_smiles", "standardized_smiles")
    for col in preferred:
        if col in df.columns:
            return col
    candidates = [c for c in df.columns if "smiles" in str(c).lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not uniquely identify SMILES column. Columns: {list(df.columns)}")


def find_target_column(df: pd.DataFrame, isoform: str) -> str | None:
    """Select point estimate, excluding confidence bounds and other uncertainty fields."""
    direct = f"{isoform}_pIC50_direct_inhibition"
    if direct in df.columns:
        return direct
    candidates = [
        c for c in df.columns
        if isoform.lower() in str(c).lower()
        and "pic50" in str(c).lower()
        and not any(token in str(c).lower() for token in
                    ("lower", "upper", "low", "high", "conf", "std", "se", "ci", "err"))
    ]
    for c in (f"{isoform}_pIC50", f"{isoform}_pic50", f"{isoform}_pIC50_mean"):
        if c in candidates:
            return c
    return candidates[0] if len(candidates) == 1 else None


def find_ci_columns(df: pd.DataFrame, target_col: str) -> tuple[str | None, str | None]:
    """Resolve the OpenADMET direct-inhibition confidence interval schema."""
    exact_lower, exact_upper = target_col + "_conf_low", target_col + "_conf_high"
    if exact_lower in df.columns and exact_upper in df.columns:
        return exact_lower, exact_upper
    prefix = target_col.split("_pIC50")[0] if "_pIC50" in target_col else target_col
    lower = [c for c in df.columns if prefix.lower() in str(c).lower()
             and any(t in str(c).lower() for t in ("conf_low", "lower", "ci_low", "ci_lower"))]
    upper = [c for c in df.columns if prefix.lower() in str(c).lower()
             and any(t in str(c).lower() for t in ("conf_high", "upper", "ci_high", "ci_upper"))]
    return (lower[0] if len(lower) == 1 else None,
            upper[0] if len(upper) == 1 else None)


def parse_molecules(smiles: Iterable[str]):
    """Return RDKit molecules; invalid structures become None."""
    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        mols.append(mol)
    return mols


def morgan_matrix(mols, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Create a dense uint8 Morgan fingerprint matrix."""
    fps = []
    for mol in mols:
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    X = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def scaffold_groups(mols) -> np.ndarray:
    """Create Bemis–Murcko scaffold keys; acyclic molecules use canonical SMILES."""
    groups = []
    for mol in mols:
        if mol is None:
            groups.append(f"INVALID_{len(groups)}")
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        key = Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else Chem.MolToSmiles(mol, canonical=True)
        groups.append(key)
    return np.asarray(groups, dtype=object)


def max_tanimoto_similarity(query_fps, reference_fps) -> np.ndarray:
    """Maximum Tanimoto similarity from each query fingerprint to reference fingerprints."""
    if not reference_fps:
        return np.zeros(len(query_fps), dtype=float)
    result = []
    for fp in query_fps:
        if fp is None:
            result.append(np.nan)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
            result.append(float(max(sims)) if sims else np.nan)
    return np.asarray(result, dtype=float)


def soft_threshold_interval_error(y_pred, ci_lower, ci_upper) -> np.ndarray:
    """Distance to a supplied interval: zero inside, distance to nearest bound outside."""
    y_pred = np.asarray(y_pred, dtype=float)
    lo = np.asarray(ci_lower, dtype=float)
    hi = np.asarray(ci_upper, dtype=float)
    err = np.maximum(lo - y_pred, 0.0) + np.maximum(y_pred - hi, 0.0)
    invalid = ~np.isfinite(y_pred) | ~np.isfinite(lo) | ~np.isfinite(hi) | (lo > hi)
    err[invalid] = np.nan
    return err


def regression_metrics(y_true, y_pred, ci_lower=None, ci_upper=None) -> dict:
    """Return basic metrics plus interval-aware error if interval bounds are available."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    result = {
        "n": int(valid.sum()),
        "mae": float(mean_absolute_error(y_true[valid], y_pred[valid])) if valid.any() else np.nan,
        "r2": float(r2_score(y_true[valid], y_pred[valid])) if valid.sum() > 1 else np.nan,
    }
    if ci_lower is not None and ci_upper is not None:
        interval_err = soft_threshold_interval_error(y_pred, ci_lower, ci_upper)
        ok = valid & np.isfinite(interval_err)
        result["mean_interval_distance"] = float(np.mean(interval_err[ok])) if ok.any() else np.nan
        result["n_with_interval"] = int(ok.sum())
    return result


def risk_coverage_curve(y_true, y_pred, uncertainty, coverages=None) -> pd.DataFrame:
    """Compute MAE among the least-uncertain fraction of predictions."""
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 10)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(uncertainty)
    yt, yp, u = y_true[valid], y_pred[valid], uncertainty[valid]
    if len(yt) == 0:
        return pd.DataFrame(columns=["coverage", "n_retained", "mae"])
    order = np.argsort(u)
    rows = []
    for coverage in coverages:
        n = max(1, int(np.ceil(float(coverage) * len(order))))
        idx = order[:n]
        rows.append({"coverage": float(coverage), "n_retained": int(n),
                     "mae": float(mean_absolute_error(yt[idx], yp[idx]))})
    return pd.DataFrame(rows)
