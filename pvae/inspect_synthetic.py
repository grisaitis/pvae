import sys
sys.path.append(".")
sys.path.append("..")

import pandas as pd
import torch
from pvae.models.utils import load_model


if __name__ == "__main__":
    device = torch.device("cpu")
    model = load_model("/home/jupyter/pvae/experiments/2024-05-29T19_39_30.1820548gbo9bj4", device)
    train_loader, test_loader = model.getDataLoaders(model.params.batch_size, True, device, *model.params.data_params)
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    for _, labels in test_loader:
        all_labels.append(labels)
    labels = torch.cat(all_labels)
    print(labels.shape)
    # df = pd.DataFrame(labels.numpy(), columns=["d_{}".format(i) for i in range(labels.shape[1])])
    # df = df.sort_values(by=list(df.columns))
    # print(df[df.columns[0]].value_counts().sort_index())
    # print(df[df.columns[:2]].value_counts().sort_index())
    # print(df[df.columns[:3]].value_counts().sort_index())
    # print(len(df))
    # print(df.sample(10))
    # print(df[(df["d_1"] < 0) & (df["d_2"] == 0)])
    labels_1d = model.map_class_labels_to_1d(labels.numpy())
    print(labels_1d)
    print(labels_1d.shape)
