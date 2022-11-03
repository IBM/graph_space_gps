import numpy as np
import json
import torch
from rdkit import Chem
import argparse

from data_prep.utils import equivalence_from_mol, ATOM_LIST


def filter_mols_by_equivalence(smiles_dict, max_equiv_size: int = 5):

    filtered_smiles = {}
    equiv_max_counts = {atom: 0 for atom in ATOM_LIST}

    for smile, y in smiles_dict.items():
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        equiv_mol = equivalence_from_mol(mol=mol)

        is_accepted = True
        for atom_label, atom_idxs in equiv_mol.items():
            if len(atom_idxs) > max_equiv_size:
                is_accepted = False
                break
        
        if is_accepted:
            counts = {label: len(atom_idxs) for label, atom_idxs in equiv_mol.items()}
            for label in counts:
                if counts[label] > equiv_max_counts[label]:
                    equiv_max_counts[label] = counts[label]
            print(max_equiv_size, counts)
            filtered_smiles[smile] = y

    print("Equivalence classes: ", max_equiv_size, equiv_max_counts)
    return filtered_smiles


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", "equiv_filter", choices=["equiv_filter", "permuted_dataset"])
    parser.add_argument("--filename")
    parser.add_argument("--max_equiv_size")


if __name__ == "__main__":
    with open("examples/data/freesolv/smiles.json", "r") as f:
        smiles_dict = json.load(f)

    for max_equiv_size in [3, 5, 7, 10]:
        filtered_smiles = filter_mols_by_equivalence(smiles_dict, max_equiv_size=max_equiv_size)

        with open(f"examples/data/freesolv/smiles_max-equiv={max_equiv_size}.json", "w") as f:
            json.dump(filtered_smiles, f)

        print(f"Number of examples: {len(filtered_smiles)}")
        print()
