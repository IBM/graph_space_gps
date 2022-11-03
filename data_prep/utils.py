import torch
from torch_geometric.data import Data
import networkx as nx
from rdkit import Chem
from torch_geometric.utils import is_undirected, to_networkx, from_networkx

ATOM_LIST = ['C', 'N', 'O', 'Cl', 'S', 'F', 'I', 'P', 'Br'] # Computed from FreeSolv


def equivalence_from_mol(mol):
    equiv_labels = {atom_type: [] for atom_type in ATOM_LIST}
    for idx, atom in enumerate(mol.GetAtoms()):
        equiv_labels[atom.GetSymbol()].append(idx)
    equiv_labels = {label: torch.tensor(indices).long() 
                    for label, indices in equiv_labels.items()}
    return equiv_labels


def edge_index_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices += [[i, j], [j, i]]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    
    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * mol.GetNumAtoms() + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
    return edge_index



def relabel_nodes(data, mapping, equiv_class):
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
    # print(rev_mapping, mapping)
    return from_networkx(G), new_equiv


def subgraph_and_equivalence(node_idx, edge_index, n_nodes, n_hops=2):
    """
    Extract k-hop subgraphs and the corresponding equivalence classes.
    k-hop neighbors are given a label k
    """
    row, col = edge_index
    
    subgraph = [row.new_tensor([node_idx]).long()]
    equiv_labels = {0: [node_idx]}
    label = 0
    
    node_mask = row.new_empty((n_nodes,), dtype=torch.bool)
    edge_mask = row.new_empty((row.size(0),), dtype=torch.bool)
    
    for hop in range(n_hops):
        node_mask.fill_(False)
        node_mask[subgraph[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        
        hop_nei = col[edge_mask].tolist()
        hop_nei = set(hop_nei).difference(set(torch.cat(subgraph).unique().tolist()))
        label += 1
        equiv_labels[label] = list(hop_nei)
        subgraph.append(col[edge_mask])
        
    subgraph = torch.cat(subgraph).unique()
    node_mask.fill_(False)
    node_mask[subgraph] = True
    edge_mask = node_mask[row] & node_mask[col]

    subgraph_edge = edge_index[:, edge_mask]
    return subgraph, subgraph_edge, equiv_labels


def compute_shifts(equiv_labels, max_counts):
    """
    Computes the shift to be applied to indices in equiv_labels.
    For each label in equiv_labels, we add dummy disconnected nodes if
    necessary, and compute the shifts for following labels accordingly.
    """
    shifts = {}
    cum_shift = 0
    
    n_nodes = sum([len(elem) for elem in equiv_labels.values()])
    shift_tensor = torch.empty((n_nodes,), dtype=torch.long)
    
    for label, indices in equiv_labels.items():
        shifts_for_label = torch.tensor([cum_shift] * len(indices)).long()
        shift_tensor[indices] = shifts_for_label
        shifts[label] = shifts_for_label

        if len(indices) != max_counts[label]:
            cum_shift += (max_counts[label] - len(indices))
    return shifts, shift_tensor


def apply_shift(edge_index, shifts_tensor):
    """
    Given shifts_tensor from the `compute_shifts` function, we apply it
    to the edge index to compute the new edge index
    """
    row, col = edge_index
    new_row = row + shifts_tensor[row]
    new_col = col + shifts_tensor[col]

    return torch.vstack([new_row, new_col])
