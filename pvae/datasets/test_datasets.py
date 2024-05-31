# module to test the JerbyArnonDataset class, using the pytest framework.

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from pvae.datasets import JerbyArnonDataset


def test_jerby_arnon_dataset():
    # Create an instance of the JerbyArnonDataset
    dataset = JerbyArnonDataset.from_csv(
        path_rnaseq="data/jerby_arnon/GSE115978_counts.csv",
        path_cell_annotations="data/jerby_arnon/GSE115978_cell.annotations.csv",
    )

    # Perform some assertions to test the dataset
    assert len(dataset) > 0  # Check if the dataset is not empty

    # Get a sample from the dataset
    data, annotations = dataset[0]

    # Perform assertions on the sample
    assert isinstance(data, torch.Tensor)  # Check if the sample is a torch.Tensor
    assert data.shape == (23686,)  # Check if the sample has the expected shape

    # Create a data loader for the dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate over the data loader
    for batch in dataloader:
        data, annotations = batch
        # Perform assertions on each batch
        assert isinstance(data, torch.Tensor)  # Check if the data is a torch.Tensor
        assert data.shape[0] == 32  # Check if the data size is correct
        for i_row in range(data.shape[0]):
            assert data[i_row].shape == (23686,)
            assert annotations["no.of.reads"][i_row] == np.sum(data[i_row]), (annotations["no.of.reads"][i_row], np.sum(data[i_row]))