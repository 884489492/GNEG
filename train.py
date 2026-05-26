import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, Set2Set
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os


def train(model, train_loader, val_loader, test_loader, train_df, num_targets, target_cols, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pos_weights = []
    for col in target_cols:
        pos = (train_df[col] == 1).sum()
        neg = (train_df[col] == 0).sum()
        if pos > 0:
            pos_weights.append(neg / pos)
        else:
            pos_weights.append(1.0)

    pos_weight = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=20, factor=0.7
    )

    best_val_auc = 0.0
    model_path = 'best_model_sider.pth'

    for epoch in range(epochs):
        model.train()
        train_preds, train_gts = [], []

        for data in train_loader:
            data = data.to(device)
            out = model(data)

            batch_size = data.num_graphs
            y = data.y.view(batch_size, num_targets)

            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            prob = torch.sigmoid(out).detach().cpu().numpy()
            train_preds.append(prob)
            train_gts.append(y.cpu().numpy())

        train_preds = np.vstack(train_preds)
        train_gts = np.vstack(train_gts)

        train_auc = 0.0
        valid_labels = 0
        for i in range(num_targets):
            if len(np.unique(train_gts[:, i])) > 1:
                train_auc += roc_auc_score(train_gts[:, i], train_preds[:, i])
                valid_labels += 1
        train_auc = train_auc / max(valid_labels, 1) if valid_labels > 0 else 0.5

        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)

                batch_size = data.num_graphs
                y = data.y.view(batch_size, num_targets)

                prob = torch.sigmoid(out).cpu().numpy()
                val_preds.append(prob)
                val_gts.append(y.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_gts = np.vstack(val_gts)

        val_auc = 0.0
        valid_labels = 0
        for i in range(num_targets):
            if len(np.unique(val_gts[:, i])) > 1:
                val_auc += roc_auc_score(val_gts[:, i], val_preds[:, i])
                valid_labels += 1
        val_auc = val_auc / max(valid_labels, 1) if valid_labels > 0 else 0.5

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train macro AUC: {train_auc:.4f} | Val macro AUC: {val_auc:.4f} | LR: {current_lr:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_preds, test_gts = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)

            batch_size = data.num_graphs
            y = data.y.view(batch_size, num_targets)

            prob = torch.sigmoid(out).cpu().numpy()
            test_preds.append(prob)
            test_gts.append(y.cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_gts = np.vstack(test_gts)

    test_auc = 0.0
    valid_labels = 0
    for i in range(num_targets):
        if len(np.unique(test_gts[:, i])) > 1:
            test_auc += roc_auc_score(test_gts[:, i], test_preds[:, i])
            valid_labels += 1
    test_auc = test_auc / max(valid_labels, 1) if valid_labels > 0 else 0.5

    print("\nFINAL TEST RESULTS (SIDER Dataset)")
    print(f" Test macro ROC-AUC: {test_auc:.4f}")

    return test_auc
