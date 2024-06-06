import logging
import typing
import pandas as pd
import torch
import torch.utils.data
import numpy as np
from csv import reader


logger = logging.getLogger(__name__)

def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_name):
        filename = 'data/{}.csv'.format(csv_name)
        dataset = np.array(load_csv(filename))
        dataset = dataset[1:, :]
        self.images = dataset[:, 0:-1].astype(np.float)
        self.latents = dataset[:, [-1]]
        self.latents = self.latents.astype(np.int)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx, :])
        latent = torch.Tensor(self.latents[idx])
        return (image, latent)


class SyntheticDataset(torch.utils.data.Dataset):
    '''
    Implementation of a synthetic dataset by hierarchical diffusion. 
    Args:
    :param int dim: dimension of the input sample
    :param int depth: depth of the tree; the root corresponds to the depth 0
    :param int :numberOfChildren: Number of children of each node in the tree
    :param int :numberOfsiblings: Number of noisy observations obtained from the nodes of the tree
    :param float sigma_children: noise
    :param int param: integer by which :math:`\\sigma_children` is divided at each deeper level of the tree
    '''
    def __init__(self, dim, depth, numberOfChildren=2, sigma_children=1, param=1, numberOfsiblings=1, factor_sibling=10):
        print("SyntheticDataset arguments:")
        import collections
        args_dict = collections.OrderedDict(
            dim=dim,
            depth=depth,
            numberOfChildren=numberOfChildren,
            sigma_children=sigma_children,
            param=param,
            numberOfsiblings=numberOfsiblings,
            factor_sibling=factor_sibling
        )
        import json
        print(json.dumps(args_dict, indent=2))
        self.dim = int(dim)
        self.root = np.zeros(self.dim)
        self.depth = int(depth)
        self.sigma_children = sigma_children
        self.factor_sibling = factor_sibling
        self.param = param
        self.numberOfChildren = int(numberOfChildren)
        self.numberOfsiblings = int(numberOfsiblings)  

        self.origin_data, self.origin_labels, self.data, self.labels = self.bst()

        # Normalise data (0 mean, 1 std)
        self.data -= np.mean(self.data, axis=0, keepdims=True)
        self.data /= np.std(self.data, axis=0, keepdims=True)

    def __len__(self):
        '''
        this method returns the total number of samples/nodes
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Generates one sample
        '''
        data, labels = self.data[idx], self.labels[idx]
        return torch.Tensor(data), torch.Tensor(labels)

    def get_children(self, parent_value, parent_label, current_depth, offspring=True):
        '''
        :param 1d-array parent_value
        :param 1d-array parent_label
        :param int current_depth
        :param  Boolean offspring: if True the parent node gives birth to numberOfChildren nodes
                                    if False the parent node gives birth to numberOfsiblings noisy observations
        :return: list of 2-tuples containing the value and label of each child of a parent node
        :rtype: list of length numberOfChildren
        '''
        if offspring:
            numberOfChildren = self.numberOfChildren
            sigma = self.sigma_children / (self.param ** current_depth)
        else:
            numberOfChildren = self.numberOfsiblings
            sigma = self.sigma_children / (self.factor_sibling*(self.param ** current_depth))
        children = []
        for i in range (numberOfChildren):
            child_value = parent_value + np.random.randn(self.dim) * np.sqrt(sigma)
            child_label = np.copy(parent_label)
            if offspring: 
                child_label[current_depth] = i + 1
            else:
                child_label[current_depth] = -i - 1
            children.append((child_value, child_label))
        return children

    def bst(self):
        '''
        This method generates all the nodes of a level before going to the next level
        '''
        queue = [(self.root, np.zeros(self.depth+1), 0)]
        visited = []
        labels_visited = []
        values_clones = []
        labels_clones = []
        while len(queue) > 0:
            current_node, current_label, current_depth = queue.pop(0)
            visited.append(current_node)
            labels_visited.append(current_label)
            if current_depth < self.depth:
                children = self.get_children(current_node, current_label, current_depth)
                for child in children:
                    queue.append((child[0], child[1], current_depth + 1)) 
            if current_depth <= self.depth:
                clones = self.get_children(current_node, current_label, current_depth, False)
                for clone in clones:
                    values_clones.append(clone[0])
                    labels_clones.append(clone[1])
        length = int(((self.numberOfChildren) ** (self.depth + 1) - 1) / (self.numberOfChildren - 1))
        length_leaves = int(self.numberOfChildren**self.depth)
        images = np.concatenate([i for i in visited]).reshape(length, self.dim)
        labels_visited = np.concatenate([i for i in labels_visited]).reshape(length, self.depth+1)[:,:self.depth]
        values_clones = np.concatenate([i for i in values_clones]).reshape(self.numberOfsiblings*length, self.dim)
        labels_clones = np.concatenate([i for i in labels_clones]).reshape(self.numberOfsiblings*length, self.depth+1)
        return images, labels_visited, values_clones, labels_clones


class JerbyArnonDataset(torch.utils.data.Dataset):
    # tpm -> 17324 genes after filters
    # counts -> 17184 after filters
    def __init__(self, df_rnaseq: pd.DataFrame, df_cell_annotations: pd.DataFrame):
        self.df_rnaseq = df_rnaseq
        self.df_cell_annotations = df_cell_annotations

    @classmethod
    def from_csv(cls, path_rnaseq: str, path_cell_annotations: str, standardize_rnaseq: bool = True):
        # note for limiting # of samples:
        # nrows=800 -> 13703 genes
        # , usecols=range(df_cell_annotations.shape[0] + 1)
        logger.debug("reading cell annotations from %s", path_cell_annotations)
        df_cell_annotations = pd.read_csv(path_cell_annotations)
        df_cell_annotations = df_cell_annotations.set_index("cells", drop=False)
        df_rnaseq = cls._read_csv_cached(path_rnaseq, sep=",", engine="c")
        df_rnaseq.columns = ["gene_symbol"] + list(df_rnaseq.columns[1:])
        df_rnaseq = df_rnaseq.set_index("gene_symbol", drop=True)
        genes = cls._get_genes_to_keep(df_rnaseq)
        cells = cls._get_cells_to_keep(df_rnaseq)
        logger.debug("shapes before filtering: %s, %s", df_rnaseq.shape, df_cell_annotations.shape)
        df_cell_annotations = df_cell_annotations.loc[cells]
        df_rnaseq = df_rnaseq.loc[genes, cells]
        logger.debug("shapes after filtering: %s, %s", df_rnaseq.shape, df_cell_annotations.shape)
        if standardize_rnaseq:
            df_rnaseq = cls._standardize_rowwise(df_rnaseq)
        assert df_rnaseq.shape[1] == df_cell_annotations.shape[0], (df_rnaseq.shape, df_cell_annotations.shape)
        return cls(df_rnaseq, df_cell_annotations)

    @staticmethod
    def _read_csv_cached(path_csv: str, **read_csv_kwargs) -> pd.DataFrame:
        path_feather = path_csv + ".feather"
        try:
            df = pd.read_feather(path_feather)
        except FileNotFoundError:
            logger.debug("reading csv from %s", path_csv)
            df_from_csv = pd.read_csv(path_csv, **read_csv_kwargs)
            logger.debug("writing feather to %s", path_feather)
            df_from_csv.to_feather(path_feather)
            df = pd.read_feather(path_feather)
            assert df_from_csv.equals(df), (df_from_csv, df)
        logger.debug("successfully read feather from %s", path_feather)
        return df

    @staticmethod
    def _standardize_rowwise(df_rnaseq: pd.DataFrame) -> pd.DataFrame:
        logger.debug("transforming gene data to z-scores, row-wise")
        # return df_rnaseq.sub(df_rnaseq.mean(axis=1), axis=0).div(df_rnaseq.std(axis=1) + 1e-8, axis=0)
        from scipy.stats import zscore
        # df_zscores = pd.DataFrame().reindex_like(df_rnaseq)
        # df_zscores[:] = zscore(df_rnaseq.values.T).T  # slow
        return pd.DataFrame(zscore(df_rnaseq.values.T).T, df_rnaseq.index, df_rnaseq.columns)

    @staticmethod
    def _standardize_columnwise(df_rnaseq: pd.DataFrame) -> pd.DataFrame:
        logger.debug("transforming cell data to z-scores, column-wise")
        from scipy.stats import zscore
        return df_rnaseq.apply(zscore)

    @staticmethod
    def _get_genes_to_keep(df_rnaseq: pd.DataFrame) -> pd.Index:
        # identify genes that are greater than zero more than 5% of the time
        too_sparse = df_rnaseq.eq(0).mean(axis=1) > 0.99
        logger.debug("identified %d too sparse genes, shape %s", too_sparse.sum(), too_sparse.shape)
        # remove mitochondrian genes
        mitochondrial = df_rnaseq.index.str.startswith("MT")
        logger.debug("identified %d mitochondrial genes", mitochondrial.sum())
        good_genes = ~(too_sparse | mitochondrial)
        # logger.debug("random sample of 100 genes: %s", np.random.default_rng(seed=0).choice(good_genes[good_genes].index, 100))
        return good_genes

    @staticmethod
    def _get_cells_to_keep(df_rnaseq: pd.DataFrame) -> pd.Index:
        # df_rnaseq: rows are genes, columns are cells
        cell_sparsity = df_rnaseq.eq(0).mean(axis=0)
        logger.debug("cell_sparsity.shape: %s", cell_sparsity.shape)
        very_sparse_single_cells = cell_sparsity > 0.9
        return ~very_sparse_single_cells

    def __len__(self):
        return len(self.df_cell_annotations)

    def _get_annotations(self, idx) -> typing.List:
        series = self.df_cell_annotations.iloc[idx]
        return series[["cells", "cell.types"]].to_list()

    def __getitem__(self, idx) -> typing.Tuple[torch.Tensor, typing.List[typing.Dict[str, typing.Any]]]:
        if torch.is_tensor(idx) or not isinstance(idx, int):
            raise TypeError("idx %s must be an integer, not %s" % (idx, type(idx)))
        rnaseq_numpy = self.df_rnaseq.iloc[:, idx].to_numpy()
        rnaseq = torch.Tensor(rnaseq_numpy)
        annotations = self._get_annotations(idx)
        return rnaseq, annotations

    def __serialize__(self):
        logger.debug("serializing JerbyArnonDataset")
        return {}

    def __deserialize__(self, state):
        logger.debug("deserializing JerbyArnonDataset")
        return JerbyArnonDataset.from_csv(
            "data/jerby_arnon/GSE115978_counts.csv",
            "data/jerby_arnon/GSE115978_cell.annotations.csv",
        )


"""
# testing in ipython

import logging
logging.basicConfig(
    level="DEBUG",
    format='%(asctime)s %(levelname)s:%(name)s:%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
from importlib import reload
import torch
import pvae.datasets.datasets as d

reload(d)
dataset = d.JerbyArnonDataset.from_csv("data/jerby_arnon/GSE115978_counts.csv", "data/jerby_arnon/GSE115978_cell.annotations.csv")

reload(d)
dataset = d.JerbyArnonDataset.from_csv("data/jerby_arnon/GSE115978_tpm.csv", "data/jerby_arnon/GSE115978_cell.annotations.csv")

reload(d)
dataset = d.JerbyArnonDataset.from_csv("data/jerby_arnon/GSE115978_counts.csv", "data/jerby_arnon/GSE115978_cell.annotations.csv")
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
data, _ = next(iter(loader))
print(data.shape)
data = data.numpy()
data.mean(axis=0), data.std(axis=0), data.mean(axis=0).shape

# make sure B cell marker genes are in dataset.df_rnaseq.index
# RPL37A, MEF2C, PLD3, and SNX2
genes_b_cell = ["RPL37A", "MEF2C", "PLD3", "SNX2"]
df_original = d.JerbyArnonDataset._read_csv_cached("data/jerby_arnon/GSE115978_tpm.csv")
df_original.loc[genes_b_cell]
dataset.df_rnaseq.loc[genes_b_cell]

import pandas as pd
df = pd.read_csv("data/jerby_arnon/GSE115978_counts.csv", nrows=10)
# rename first column to "gene_symbol"
df.columns = ["gene_symbol"] + list(df.columns[1:])

reload(d)
dataset = d.JerbyArnonDataset.from_csv(
    "data/jerby_arnon/GSE115978_counts.csv",
    "data/jerby_arnon/GSE115978_cell.annotations.csv",
    standardize_rnaseq=False,
)
dataset.df_rnaseq
data, _ = dataset[0]
# what is the dtype of data?
data
"""