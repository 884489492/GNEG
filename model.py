import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, Set2Set


class GNEG(nn.Module):
    def __init__(self, node_dim, edge_dim=6, hidden_dim=128, num_layers=5, num_classes=1):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.global_node = nn.Embedding(1, hidden_dim)
        self.gn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.set2set = Set2Set(hidden_dim, processing_steps=6)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)
        gn = self.global_node.weight.repeat(data.num_graphs, 1)

        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, data.edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            gn = gn + self.gn_mlp(global_mean_pool(h, data.batch))
            h = h + gn[data.batch]

        h_mean = global_mean_pool(h, data.batch)
        h_max = global_max_pool(h, data.batch)
        h_set = self.set2set(h, data.batch)
        h_gn = gn
        h = torch.cat([h_mean, h_max, h_set, h_gn], dim=-1)

        return self.classifier(h)
