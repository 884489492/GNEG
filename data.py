import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
import numpy as np


def one_hot(x, allowable):
    return [int(x == s) for s in allowable]


def atom_features(atom):
    return torch.tensor(
        one_hot(atom.GetAtomicNum(), list(range(1, 21))) +
        one_hot(atom.GetHybridization(), [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3
        ]) + [
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetIsAromatic(),
            atom.IsInRing(),
            atom.GetMass() / 100.0
        ],
        dtype=torch.float32
    )


def bond_features(bond):
    bt = bond.GetBondType()
    return torch.tensor([
        bt == rdchem.BondType.SINGLE,
        bt == rdchem.BondType.DOUBLE,
        bt == rdchem.BondType.TRIPLE,
        bt == rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ], dtype=torch.float32)


class SiderDataset(Dataset):
    def __init__(self, df, target_cols):
        self.graphs = []
        invalid_count = 0
        self.target_cols = target_cols

        for _, row in df.iterrows():
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                continue

            x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
            edge_index, edge_attr = [], []

            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = bond_features(bond)
                edge_index += [[i, j], [j, i]]
                edge_attr += [bf, bf]

            if len(edge_index) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 6), dtype=torch.float32)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.stack(edge_attr)

            y_values = row[self.target_cols].values.astype(np.float32)
            y = torch.tensor(y_values, dtype=torch.float32)

            self.graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class BBBPDataset(Dataset):
    def __init__(self, df, target_col='p_np'):
        self.graphs = []
        self.smiles_list = df['smiles'].tolist()
        invalid_count = 0
        for idx, row in df.iterrows():
            smiles = row['smiles']
            label = row[target_col]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                continue

            x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
            edge_index, edge_attr = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bf = bond_features(bond)
                edge_index += [[i, j], [j, i]]
                edge_attr += [bf, bf]

            if len(edge_index) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 6), dtype=torch.float32)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.stack(edge_attr)

            y = torch.tensor([float(label)], dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.smiles = smiles
            self.graphs.append(data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def load_data(batch_size=64):
    df = pd.read_csv('/kaggle/input/dataset123/dataset/sider.csv')

    target_cols = df.columns[1:]
    num_targets = len(target_cols)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        SiderDataset(train_df, target_cols),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Batch.from_data_list
    )
    val_loader = DataLoader(
        SiderDataset(val_df, target_cols),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Batch.from_data_list
    )
    test_loader = DataLoader(
        SiderDataset(test_df, target_cols),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Batch.from_data_list
    )

    return train_loader, val_loader, test_loader, train_df, num_targets, target_cols


def get_bbbp_train_loader():
    df = pd.read_csv('/kaggle/input/dataset123/dataset/bbbp.csv')
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df['p_np'], random_state=42)

    train_loader = DataLoader(
        BBBPDataset(train_df),
        batch_size=32,
        shuffle=True
    )
    return train_loader
