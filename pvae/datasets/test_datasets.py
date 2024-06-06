# module to test the JerbyArnonDataset class, using the pytest framework.

import argparse
import json
import pandas as pd
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from pvae.datasets import JerbyArnonDataset
from pvae.models.tabular import Rnaseq



def load_model_params(run_path: str) -> argparse.Namespace:
    run_path_args_json = run_path + '/args.json'
    with open(run_path_args_json, 'r') as f:
        run_path_args_dict = json.load(f)
    run_path_args = argparse.Namespace(**run_path_args_dict)
    return run_path_args


def test_jerby_arnon_dataset():
    # Create an instance of the JerbyArnonDataset
    dataset = JerbyArnonDataset.from_csv(
        path_rnaseq="data/jerby_arnon/GSE115978_counts.csv",
        path_cell_annotations="data/jerby_arnon/GSE115978_cell.annotations.csv",
    )

    # Perform some assertions to test the dataset
    assert len(dataset) > 0  # Check if the dataset is not empty

    # Check cell order consistency
    data_df_0 = dataset._get_rnaseq_df(0)
    anno_0 = dataset._get_annotations(0)
    assert data_df_0.columns[0] == anno_0[0][0], (data_df_0.columns[0], anno_0[0][0])
    data_df_100 = dataset._get_rnaseq_df(slice(100))
    anno_100 = dataset._get_annotations(slice(100))
    # assert data_df_100.columns.equals() == [a["cells"] for a in anno_100], (data_df_100.columns, [a["cells"] for a in anno_100])
    pd.testing.assert_index_equal(data_df_100.columns, pd.Index([a[0] for a in anno_100]))

    # Get a sample from the dataset
    data_0, _ = dataset[0]
    assert isinstance(data_0, torch.Tensor)  # Check if the sample is a torch.Tensor
    assert data_0.shape == (23686,)  # Check if the sample has the expected shape

    # Create a data loader for the dataset
    BATCH_SIZE = 32
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get a batch from the data loader
    data, annotations = next(dataloader)
    assert isinstance(data, torch.Tensor)  # Check if the data is a torch.Tensor
    assert data.shape == (BATCH_SIZE, 23686), data.shape
    for i_row in range(data.shape[0]):
        assert data[i_row].shape == (23686,)
        # assert annotations["no.of.reads"][i_row] == np.sum(data[i_row]), (annotations["no.of.reads"][i_row], np.sum(data[i_row]))
        
def test_enumerate_data_loader():
    # args = load_model_params("experiments/2024-06-03T21_31_03.122060cyyqhmge")
    # rnaseq_model.getDataLoaders(args.batch_size, True, "cpu", *args.data_params)
    BATCH_SIZE = 32
    train_loader, test_loader = Rnaseq.getDataLoaders(BATCH_SIZE, True, "cpu")
    for i_batch, (data, labels) in enumerate(train_loader):
        assert data.shape[0] == BATCH_SIZE, data.shape
        assert len(labels) == BATCH_SIZE, len(labels)
