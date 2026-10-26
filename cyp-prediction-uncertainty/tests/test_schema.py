import numpy as np
import pandas as pd
from rdkit import Chem
from src.cyp_uncertainty import (
    find_smiles_column, find_target_column, find_ci_columns,
    scaffold_groups, morgan_matrix, regression_metrics, soft_threshold_interval_error,
)


def test_openadmet_column_mapping_exact_schema():
    isoforms = ('CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4')
    columns = ['Molecule_Name', 'SMILES']
    for iso in isoforms:
        base = f'{iso}_pIC50_direct_inhibition'
        columns += [base, f'{base}_conf_high', f'{base}_conf_low', f'{base}_std']
    # Include bound columns before point-estimate columns, as a robustness check.
    frame = pd.DataFrame(columns=list(reversed(columns)))
    assert find_smiles_column(frame) == 'SMILES'
    for iso in isoforms:
        base = f'{iso}_pIC50_direct_inhibition'
        assert find_target_column(frame, iso) == base
        assert find_ci_columns(frame, base) == (f'{base}_conf_low', f'{base}_conf_high')


def test_invalid_intervals_are_not_scored():
    got = soft_threshold_interval_error([3., 3., 3.], [2., 5., np.nan], [4., 4., 4.])
    assert got[0] == 0
    assert np.isnan(got[1])
    assert np.isnan(got[2])
    metrics = regression_metrics([2., 3.], [2.3, 2.8], [1.8, 2.9], [2.5, 3.1])
    assert metrics['mean_interval_distance'] > 0
    assert metrics['n_with_interval'] == 2


def test_morgan_and_scaffolds():
    mols = [Chem.MolFromSmiles(s) for s in ('c1ccccc1', 'Cc1ccccc1', 'c1ccncc1')]
    x = morgan_matrix(mols, n_bits=128)
    groups = scaffold_groups(mols)
    assert x.shape == (3, 128)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
