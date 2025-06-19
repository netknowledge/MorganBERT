# Adapted based on https://github.com/chemprop/chemprop

import random

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_split(smiles_series=None, mol_series=None, sizes=(0.8,0.2), balanced=True, include_chirality=True, seed=0):
    """
    Parameters
    ----------
    smiles_series : Pandas Series 
    mol_series : Pandas Series
    balanced : bool
        True: put index larger than the threshold (half of test set) at first
        False: sort by scaffold size

    Returns
    -------
    index to `.loc` the data
    """
    assert smiles_series is not None or mol_series is not None
    assert sum(sizes) == 1
    mols = mol_series if mol_series is not None else smiles_series.apply(Chem.MolFromSmiles)
    N = len(mols)
    train_size = sizes[0] * N
    test_size = sizes[-1] * N
    len_th = test_size / 2
    # ordered list of index
    scaffold_to_idx = defaultdict(list)
    for idx, mol in mols.items():
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffold_to_idx[scaffold].append(idx)
    if balanced:
        big_index, small_index = [], []
        for index in scaffold_to_idx.values():
            if len(index) > len_th:
                big_index.append(index)
            else:
                small_index.append(index)
        random.seed(seed)
        random.shuffle(big_index)
        random.shuffle(small_index)
        index_list = big_index + small_index
    else:
        index_list = sorted(list(scaffold_to_idx.values()), key=lambda x: len(x), reverse=True)
    # split
    if len(sizes) == 2: # train test split
        train, test = [], []
        for index in index_list:
            if len(train) + len(index) <= train_size:
                train.extend(index)
            else:
                test.extend(index)
        return train, test
    elif len(sizes) == 3: # train valid test split
        train, valid, test = [], [], []
        valid_cutoff = sum(sizes[:2]) * N
        for index in index_list:
            if len(train) + len(index) > train_size:
                if len(train) + len(valid) + len(index) > valid_cutoff:
                    test.extend(index)
                else:
                    valid.extend(index)
            else:
                train.extend(index)
        return train, valid, test
