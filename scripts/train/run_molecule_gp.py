import enum
import torch
import argparse
from rdkit import Chem
import gpytorch
from typing import List
import numpy as np
import json
import warnings
from torch_geometric.data import Data

from gpytorch.constraints import Interval
from torch_geometric.utils import is_undirected, to_networkx
import networkx as nx

warnings.filterwarnings("ignore")

from projected.kernels.utils import build_kernel
from data_prep.utils import (equivalence_from_mol,
    edge_index_from_smiles, compute_shifts, apply_shift, ATOM_LIST)


def relabel_mol_edge_index(data, mapping, equiv_class):
    # Mapping is a dict from old_label to new_label
    undirected = is_undirected(data.edge_index)
    G = to_networkx(data=data, to_undirected=undirected)
    G = nx.relabel_nodes(G, mapping=mapping)
    rev_mapping = {value: idx for idx, (key, value) in enumerate(mapping.items())}
    
    n_inputs = 0
    new_equiv = {}
    for key, value in equiv_class.items():
        new_equiv[key] = []
        for elem in value:
            new_equiv[key].append(n_inputs)
            n_inputs += 1
    
    G = nx.relabel_nodes(G, mapping=rev_mapping)
    G = G.to_directed()
    
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return edge_index, new_equiv


class MoleculeGP(gpytorch.models.ExactGP):

    def __init__(self, kernel, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_freesolv(filename, use_equiv: bool = False):
    with open(f"{filename}", "r") as f:
        splits = json.load(f)

    train_smiles_y = splits['train']
    test_smiles_y = splits['test']

    if use_equiv:
        equiv_class_train, equiv_class_test = [], []
        counts = {atom: 0 for atom in ATOM_LIST}

    mol_size = 0
    for smile in train_smiles_y:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            if mol.GetNumAtoms() > mol_size:
                mol_size = mol.GetNumAtoms()
    
    for smile in test_smiles_y:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            if mol.GetNumAtoms() > mol_size:
                mol_size = mol.GetNumAtoms()

    X_train, y_train = [], []
    X_test, y_test = [], []

    data_train, data_test = [], []
    for smile, y_val in train_smiles_y.items():
        mol = Chem.MolFromSmiles(smile)

        if mol.GetNumAtoms() == 1:
            continue
        
        if use_equiv:
            equiv_labels = equivalence_from_mol(mol)
            for key, value in counts.items():
                if len(equiv_labels[key]) > value:
                    counts[key] = len(equiv_labels[key])

        edge_index = edge_index_from_smiles(smile)
        tmp_data = Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())
        if use_equiv:
            equiv_class_train.append(equiv_labels)
        data_train.append(tmp_data)
        y_train.append(y_val)

    for smile, y_val in test_smiles_y.items():
        mol = Chem.MolFromSmiles(smile)

        if mol.GetNumAtoms() == 1:
            continue
        
        if use_equiv:
            equiv_labels = equivalence_from_mol(mol)
            for key, value in counts.items():
                if len(equiv_labels[key]) > value:
                    counts[key] = len(equiv_labels[key])

        edge_index = edge_index_from_smiles(smile)
        tmp_data = Data(edge_index=edge_index, num_nodes=mol.GetNumAtoms())
        if use_equiv:
            equiv_class_test.append(equiv_labels)
        data_test.append(tmp_data)
        y_test.append(y_val)

    if use_equiv:
        n = sum(counts.values())
        print(f"Size of equivalence classes: {counts}")
    else:
        n = mol_size
    d = n * n

    for idx, tmp_data in enumerate(data_train):
        if use_equiv:
            equiv_class = equiv_class_train[idx]
            mapping = {}
            for _, value in equiv_class.items():
                for elem in value:
                    mapping[elem.item()] = chr(ord("a") + len(mapping))
        
        if use_equiv:
            new_edge_index, new_equiv = relabel_mol_edge_index(data=tmp_data, mapping=mapping, equiv_class=equiv_class)
            shifts, shift_tensor = compute_shifts(new_equiv, counts)
            new_edge_index = apply_shift(new_edge_index, shifts_tensor=shift_tensor)
            row, col = new_edge_index
        else:
            row, col = tmp_data.edge_index
        adj = torch.zeros((n,n), dtype=torch.float)
        adj[row, col] = 1
        adj = adj.flatten().long()
        X_train.append(adj)
    
    for idx, tmp_data in enumerate(data_test):
        if use_equiv:
            equiv_class = equiv_class_test[idx]
            mapping = {}
            for _, value in equiv_class.items():
                for elem in value:
                    mapping[elem.item()] = chr(ord("a") + len(mapping))
        
        if use_equiv:
            new_edge_index, new_equiv = relabel_mol_edge_index(data=tmp_data, mapping=mapping, equiv_class=equiv_class)
            shifts, shift_tensor = compute_shifts(new_equiv, counts)
            new_edge_index = apply_shift(new_edge_index, shifts_tensor=shift_tensor)
            row, col = new_edge_index
        else:
            row, col = tmp_data.edge_index
        adj = torch.zeros((n,n), dtype=torch.float)
        adj[row, col] = 1
        adj = adj.flatten().long()
        X_test.append(adj)

    if use_equiv:
        equiv = []
        prev = 0
        for label, count in counts.items():
            equiv.append(list(range(prev, prev + count)))
            prev += count
        return (X_train, y_train), (X_test, y_test), equiv, n
    return (X_train, y_train), (X_test, y_test), n


def build_gaussian_process(args, train_inputs, n: int, equiv_classes=None):
    X_train, y_train = train_inputs
    base_kernel = build_kernel(args, n=n, equiv_classes=equiv_classes, 
                          is_undirected=True, classification=False)

    kernel = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_constraint=None)
    kernel.outputscale = args.sigma2   

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=None)
    likelihood.noise = args.lik_var
    model = MoleculeGP(kernel=kernel, X_train=X_train, 
                      y_train=y_train, likelihood=likelihood)
    return model, likelihood


def train_loop(args, train_inputs, test_inputs, model, likelihood, orig_std):
    model.train()
    likelihood.train()

    X_train, y_train = train_inputs
    X_test, y_test = test_inputs

    optimizer = torch.optim.Adam([param for name, param in model.named_parameters()], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)  

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(args.train_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train).mean()
        loss.backward()
        norm_params = {name: p.grad.detach()for name, p in model.named_parameters() if p.grad is not None}
        norm_all = torch.norm(torch.tensor(list(norm_params.values())))
        optimizer.step()

        if i % args.print_every == 0:
            print()
            print(f"After Iter {i}/{args.train_iter}", flush=True)
            print_msg = "Train Parameters ||"
            print_msg += f"Loss: {np.round(loss.item(), 4)}"

            kappa = model.covar_module.base_kernel.kappa.item()
            print_msg += f" | Kappa: {np.round(kappa, 4)}"
            
            sigma2 = model.covar_module.outputscale.item()
            print_msg += f" | Sigma2: {np.round(sigma2, 4)}"
            
            if hasattr(model.covar_module.base_kernel, "nu"):
                nu = model.covar_module.base_kernel.nu.item()
                print_msg += f" | Nu: {np.round(nu, 4)}"
            
            print_msg += f" | Noise: {np.round(model.likelihood.noise.item(), 4)}"
            print(print_msg, flush=True)

        if i % args.eval_every == 0:
            model.eval()
            likelihood.eval()

            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_y = likelihood(model(X_test))
                pred_f = model(X_test)
                l2_error = orig_std * torch.linalg.norm(y_test - pred_y.mean) / len(y_test)
                l2_error_base = orig_std * torch.linalg.norm(y_test) / len(y_test)

                l2_error = np.round(l2_error.item(), 4)
                l2_error_base = np.round(l2_error_base.item(), 4)
                lik_test_y = np.round(pred_y.log_prob(y_test).item(), 4)
                lik_test_f = np.round(pred_f.log_prob(y_test).item(), 4)

                print(f"Test Metrics || Likelihood-Y {lik_test_y} | Likelihood-F {lik_test_f} | L2 Error: {l2_error} | L2 Norm Naive: {l2_error_base}")
                #model_save_file = f"freesolv-kernel={args.kernel}-mode={args.kernel_mode}_lr={args.lr}_"
                #if args.kernel == "graph_quotient":
                #    model_save_file += f"approx-count={args.n_approx}_"
                #model_save_file += "model"
                #torch.save(model.state_dict(), f"models/{model_save_file}.pth")
            
            model.train()
            likelihood.train()

        if args.step_every is not None:
            if i % args.step_every == 0 and i:
                scheduler.step()
                print("Learning rate after stepping is: ", scheduler.get_last_lr())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_equiv_class", action='store_true')
    parser.add_argument("--filename")

    parser.add_argument("--trainable_params", default=None, nargs="+")
    parser.add_argument('--kernel', default="graph", 
                        help="Which kernel to use. Choices: [graph, graph_quotient]")
    parser.add_argument("--kernel_mode", default="heat", help="Mode of kernel to use. Choices: [heat, matern, additive]")
    parser.add_argument("--kappa", default=1.0, type=float, help="Initial value of kappa")
    parser.add_argument("--sigma2", default=1.0, type=float, help="Initial value of sigma2")
    parser.add_argument("--nu", default=2.5, type=float, help="Initial value of nu. Only used with matern kernel")
    parser.add_argument("--n_approx", default=0, type=int)

    parser.add_argument("--lik_var", default=0.01, type=float)

    parser.add_argument("--train_iter", default=1000, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--print_every", default=500, type=int)
    parser.add_argument("--step_every", default=None, type=int)

    args = parser.parse_args()
    print(f"Args: {args}")
    return args  


def main():
    args = parse_args()
    
    if args.use_equiv_class:
        train_inputs, test_inputs, equiv_classes, n = load_freesolv(filename=args.filename, use_equiv=True)
    else:
        train_inputs, test_inputs, n = load_freesolv(filename=args.filename, use_equiv=False)
        equiv_classes = None

    X_train, y_train = train_inputs
    X_train = torch.stack(X_train).long()
    y_train = torch.tensor(y_train).flatten()

    X_test, y_test = test_inputs
    X_test = torch.stack(X_test).long()
    y_test = torch.tensor(y_test).flatten()

    orig_mean, orig_std = torch.mean(y_train), torch.std(y_train)
    y_train = (y_train-orig_mean)/orig_std
    y_test = (y_test-orig_mean)/orig_std

    print(f"Train x: {X_train.shape}, Train y: {y_train.shape}")
    print(f"Test x: {X_test.shape}, Test y: {y_test.shape}")
    print(f"Mean y: {orig_mean}, std y: {orig_std}")

    model, likelihood = build_gaussian_process(args=args, train_inputs=(X_train, y_train), 
                                               n=n, equiv_classes=equiv_classes)
    train_loop(args, (X_train, y_train), (X_test, y_test), model, likelihood, orig_std)


if __name__ == "__main__":
    main()
