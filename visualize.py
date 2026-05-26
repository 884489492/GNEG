import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, Set2Set
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO
from PIL import Image
from data import get_bbbp_train_loader
from model import GNEG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = get_bbbp_train_loader()

model_path = '/kaggle/working/bbbp.pth'
sample_batch = next(iter(train_loader))
node_dim = sample_batch.x.size(1)

model = GNEG(node_dim=node_dim, num_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

mols = []
legends = []
highlight_colors = []
permeable_collected = 0
non_permeable_collected = 0
max_per_class = 3

with torch.no_grad():
    for batch in train_loader:
        if permeable_collected >= max_per_class and non_permeable_collected >= max_per_class:
            break

        batch = batch.to(device)
        x = model.node_emb(batch.x)
        edge_attr = model.edge_emb(batch.edge_attr)
        vn = model.virtual_node.weight.repeat(batch.num_graphs, 1)

        layer_activations = []
        h = x
        for conv, bn in zip(model.convs, model.bns):
            h = conv(h, batch.edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            layer_activations.append(h.norm(dim=-1).cpu())
            vn = vn + model.vn_mlp(global_mean_pool(h, batch.batch))
            h = h + vn[batch.batch]

        h_mean = global_mean_pool(h, batch.batch)
        h_max = global_max_pool(h, batch.batch)
        h_set = model.set2set(h, batch.batch)
        h_vn = vn
        h_graph = torch.cat([h_mean, h_max, h_set, h_vn], dim=-1)
        logits = model.classifier(h_graph).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        ptr = 0
        for i in range(batch.num_graphs):
            if permeable_collected >= max_per_class and non_permeable_collected >= max_per_class:
                break

            n_atoms = batch[i].x.size(0)
            imp_layers = [act[ptr:ptr + n_atoms] for act in layer_activations]
            importance = torch.stack(imp_layers).mean(dim=0).numpy()
            ptr += n_atoms

            smiles = batch[i].smiles
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            true_label = int(batch[i].y.item())
            pred_prob = probs[i]
            pred_label = 1 if pred_prob > 0.5 else 0

            confidence = abs(pred_prob - 0.5)
            if confidence < 0.3:
                continue

            norm = Normalize(vmin=importance.min(), vmax=importance.max() or 1e-8)
            colors = {j: plt.cm.Reds(norm(val))[:3] for j, val in enumerate(importance)}

            legend = f"True: {true_label} (Permeable) | Pred: {pred_label}\nProb: {pred_prob:.3f}"

            if true_label == 1 and permeable_collected < max_per_class:
                mols.append(mol)
                legends.append(legend)
                highlight_colors.append(colors)
                permeable_collected += 1
            elif true_label == 0 and non_permeable_collected < max_per_class:
                mols.append(mol)
                legends.append(legend)
                highlight_colors.append(colors)
                non_permeable_collected += 1

if len(mols) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(len(mols)):
        mol = mols[i]
        legend = legends[i]
        colors = highlight_colors[i]

        atom_colors = {}
        for atom_idx, color in colors.items():
            atom_colors[atom_idx] = color

        drawer = Draw.MolDraw2DCairo(600, 600)
        drawer.drawOptions().useBWAtomPalette()

        highlight_atoms = list(colors.keys())
        highlight_atom_colors = {k: colors[k] for k in highlight_atoms}

        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_atom_colors
        )
        drawer.FinishDrawing()

        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))

        axes[i].imshow(img)
        axes[i].set_title(legend, fontsize=12)
        axes[i].axis('off')

    for j in range(len(mols), 6):
        axes[j].axis('off')

    plt.suptitle('GNEG Interpretability in BBBP\n'
                 'Redder atoms = higher contribution via global proxy node (long-range dependency)',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
