from concurrent.futures import process
import numpy as np
import torch
from rdkit import Chem
import random
import json
import argparse


def prepare_splits_freesolv(filename, 
                            split: str = 'random', 
                            train_f: float = 0.8, 
                            test_f: float = 0.2, 
                            allowed_atoms = set(['C', 'N', 'O', 'Cl']),
                            **kwargs):
    with open(filename, "r") as f:
        smiles_dict = json.load(f)

    print(f"Allowed atoms: {list(allowed_atoms)}")

    filtered_smiles = []
    for smile in smiles_dict:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            atom_set = set([atom.GetSymbol() for atom in mol.GetAtoms()])
            # The atom set from the molecule should not have any atoms besides those in the allowed set
            if not len(atom_set.difference(allowed_atoms)):
                filtered_smiles.append(smile)

    if split == "random":
        n_train = int(len(filtered_smiles) * train_f)
        n_test = int(len(filtered_smiles) * test_f)
        random.shuffle(filtered_smiles)

        train_smiles = {smile: smiles_dict[smile] for smile in filtered_smiles[: n_train]}
        test_smiles = {smile: smiles_dict[smile] for smile in filtered_smiles[n_train: n_train + n_test]}

    splits = {'train': train_smiles, 'test': test_smiles}
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")

    atom_print = "-".join(atom for atom in list(allowed_atoms))
    save_file = filename.split(".")[0] + f"_atoms-{atom_print}-{split}_split.json"

    with open(save_file, "w") as f:
        json.dump(splits, f)


SPLIT_FNS = {"freesolv": prepare_splits_freesolv}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='freesolv')
    parser.add_argument("--filename")
    parser.add_argument("--split", type=str, default="random")

    parser.add_argument("--train_f", type=float, default=0.8)
    parser.add_argument("--test_f", type=float, default=0.2)
    
    parser.add_argument("--allowed_atoms", nargs="+", default=["C", "N", "O"])

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    split_fn = SPLIT_FNS.get(args.dataset)
    print(f"Preparing {args.split} for {args.dataset}...", flush=True)

    kwargs = {}
    if args.dataset == "freesolv":
        kwargs['allowed_atoms'] = set(args.allowed_atoms)

    split_fn(filename=args.filename, split=args.split, 
              train_f=args.train_f, test_f=args.test_f, **kwargs)
    print("Done.")


if __name__ == "__main__":
    main()
