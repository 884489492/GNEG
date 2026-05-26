from model import GNEG
from train import train
from data import load_data


if __name__ == '__main__':
    train_loader, val_loader, test_loader, train_df, num_targets, target_cols = load_data()

    sample = next(iter(train_loader))
    node_dim = sample.x.size(1)

    model = GNEG(node_dim=node_dim, num_classes=num_targets)
    test_auc = train(model, train_loader, val_loader, test_loader, train_df, num_targets, target_cols, epochs=200)

    print(f"\nFinal Test macro ROC-AUC on SIDER: {test_auc:.4f}")
