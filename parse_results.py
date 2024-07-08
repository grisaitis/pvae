import argparse
import json
import logging
from pathlib import Path
from pprint import pprint
from typing import Dict, List
import warnings

import pandas as pd
import torch
import numpy as np
import torch.jit

logger = logging.getLogger(__name__)


class ExperimentOutput:
    def __init__(self, path: Path):
        self.path = path
        self.args = torch.load(self.path / "args.rar")  # returns argparse.Namespace
        self.args_json = json.load(open(self.path / "args.json"))
        # try:
        #     self.args = argparse.Namespace(**self.args_json)
        # except RuntimeError:
        #     self.args = argparse.Namespace(**self.args_json)
        assert type(self.args) is argparse.Namespace
        # load python version used, pytorch, commit hash of this repo, and other runtime info

    @classmethod
    def from_experiment_dir(cls, experiment_dir: str):
        return cls(Path("experiments") / experiment_dir)

    def started(self) -> bool:
        return (self.path / "losses.rar").exists()

    def stopped(self) -> bool:
        with open(self.path / "run.log") as f:
            # "[ME-VAE]" in one of last 5 lines of log file
            return any("[ME-VAE]" in line for line in f.readlines()[-5:])

    def complete(self):
        if not (self.path / "losses.rar").exists():
            return False
        if not self.stopped():
            return False
        experiment_epochs = self.args_json["epochs"]
        if (
            self.args_json["model"] == "mnist"
            and not (self.path / "recon_{:03d}.png".format(experiment_epochs)).exists()
        ):
            return False
        return True

    def get_epoch_time_durations(self) -> Dict[int, float]:
        # parse run.log, extracting values
        # lines with data look like "====>            Time: 0.47 s"
        with open(self.path / "run.log") as f:
            lines = f.readlines()
        lines_with_time = filter(lambda x: "====>            Time: " in x, lines)
        durations = map(lambda x: float(x.split(" ")[-2]), lines_with_time)
        return list(durations)

    def _load_losses_rar(self) -> Dict[str, list]:
        # keep this method for type annotation
        logger.debug("Loading losses from %s", self.path)
        return torch.load(self.path / "losses.rar")

    def get_loss_df(self) -> pd.DataFrame:
        losses = self._load_losses_rar()
        # print length of all lists or arrays in losses, which is a dictionary
        args_to_include = {
            k: v
            for k, v in vars(self.args).items()
            if not isinstance(v, (list, dict)) or k == "data_size"
        }
        nan_value = np.nan
        # try:
        #     args_to_include["data_size"] = args_to_include["data_size"][0]
        # except IndexError:
        #     logger.warning("data_size (%s) was empty, so setting to %s", args_to_include["data_size"], str(nan_value))
        #     args_to_include["data_size"] = nan_value
        args_to_include["data_size"] = str(args_to_include["data_size"])
        n_train_losses = len(losses["train_loss"])
        for loss_key in ["test_loss", "test_mlik", "test_recon", "test_kl"]:
            if loss_key not in losses:
                losses[loss_key] = [nan_value] * n_train_losses
            elif isinstance(losses[loss_key], list):
                save_freq = args_to_include["save_freq"]
                if len(losses[loss_key]) == 0:
                    losses[loss_key] = [nan_value] * n_train_losses
                elif len(losses[loss_key]) == 1:
                    assert isinstance(losses[loss_key][0], (int, float)), losses[
                        loss_key
                    ]
                    losses[loss_key] = losses[loss_key] * n_train_losses
                elif len(losses[loss_key]) == n_train_losses:
                    pass
                elif save_freq > 1 and save_freq < n_train_losses:
                    new_values = [nan_value] * n_train_losses
                    new_values[save_freq - 1 :: save_freq] = losses[loss_key]
                    losses[loss_key] = new_values
                else:
                    raise ValueError(
                        "Unexpected length of losses[k]: %d", len(losses[loss_key])
                    )
            else:
                raise ValueError(
                    "Unexpected type of losses[k]: %s", type(losses[loss_key])
                )
        data = {
            "experiment_dir": self.path.name,
            "started": self.started(),
            "stopped": self.stopped(),
            "complete": self.complete(),
            "epoch": list(range(1, n_train_losses + 1)),
            "test_mlik": losses["test_mlik"],
            "test_loss": losses["test_loss"],
            "test_recon": losses["test_recon"],
            "test_kl": losses["test_kl"],
            "train_loss": losses["train_loss"],
            "train_recon": losses["train_recon"],
            "train_kl": losses["train_kl"],
            **args_to_include,
        }
        try:
            df = pd.DataFrame(data)
        except ValueError:
            for loss_key, x in data.items():
                if isinstance(x, list):
                    logger.debug("data[%s] length: %d", loss_key, len(x))
                else:
                    # logger.debug("data[%s], not a list: %s", k, str(x))
                    pass
            logger.debug(
                "data array lengths: \n%s",
                json.dumps(
                    {k: len(x) for k, x in data.items()}, indent=2, sort_keys=True
                ),
            )
            raise
        for loss_key in losses.keys():
            # replace extreme values with NaN
            df[loss_key] = df[loss_key].apply(
                lambda x: x if x is not None and 0 < x < 5e4 else None
            )
        df[list(losses.keys())] = df[list(losses.keys())].astype(float)
        df["is_completion_epoch"] = df["epoch"] == df["epochs"]
        df["is_last_epoch"] = df["epoch"] == df["epoch"].max()
        epoch_durations = self.get_epoch_time_durations()
        if len(epoch_durations) < len(df):
            epoch_durations += [np.nan] * (len(df) - len(epoch_durations))
        df["epoch_duration"] = epoch_durations
        df["epoch_duration_median"] = df["epoch_duration"].median()
        return df


def get_started_experiments(path: Path) -> List[ExperimentOutput]:
    experiments = []
    for p in sorted(path.iterdir()):
        if not p.is_dir():
            continue
        try:
            e = ExperimentOutput(p)
            if e.started():
                experiments.append(e)
        except (FileNotFoundError, RuntimeError):
            continue
    return experiments


def combine_loss_dfs(experiments: List[ExperimentOutput]) -> pd.DataFrame:
    # df_dict = {e.path.name: e.make_loss_df() for e in experiments}
    # return pd.concat(df_dict, names=["experiment_dir"])
    return pd.concat([e.get_loss_df() for e in experiments], ignore_index=True)


def get_last_losses(results: pd.DataFrame) -> pd.DataFrame:
    df = results[results["is_last_epoch"]]
    cols = [
        "experiment_dir",
        "manifold",
        "c",
        "posterior",
        "dec",
        "epochs",
        # "started",
        "stopped",
        # "complete",
        "epoch",
        "train_loss",
        "train_recon",
        "train_kl",
        "test_loss",
        "test_recon",
        "test_kl",
        "test_mlik",
        # "epoch_duration_median",
    ]
    return df[cols]


def summarize_complete_experiments(df_losses_completed: pd.DataFrame) -> pd.DataFrame:
    groupbys = [
        "model",
        "data_size",
        "manifold",
        "posterior",
        "K",
        "c",
        "dec",
        "hidden_dim",
        # "lr",
        "prior_std",
        "enc",
        "dec",
        # "batch_size",
        "epochs",
        # "iwae_samples",
        # "seed",
        "stopped",
    ]

    def std_pop(x):
        return x.std(ddof=0)

    def mean_pom_95ci(x):
        mean = x.mean()
        ci = 1.96 * x.std()
        n = x.count()
        return f"{mean:.1f} ± {ci:05.1f}"

    aggs = {
        # "train_loss": [mean_pom_95ci],
        # "train_recon": [mean_pom_95ci],
        # "train_kl": [mean_pom_95ci],
        # "test_loss": ["median", mean_pom_95ci],
        "test_mlik": [mean_pom_95ci, "count"],
        # "test_recon": ["median", mean_pom_95ci],
        # "test_kl": ["count", "mean", "std"],
        # "epoch_duration_median": ["mean"],
        "experiment_dir": [max],
    }
    dfg = df_losses_completed.groupby(groupbys)
    try:
        aggregations = dfg.agg(aggs)
    except pd.core.base.DataError:
        print(dfg[list(aggs.keys())].dtypes)
        raise
    return aggregations


def summarize_started_experiments(results: pd.DataFrame) -> pd.DataFrame:
    groupbys = [
        "model",
        "manifold",
        "posterior",
        "c",
        "dec",
        "data_size",
        "batch_size",
        "stopped",
    ]
    aggs = {
        "complete": ["mean"],
        "epoch": ["mean"],
        "experiment_dir": ["nunique"],
        "epoch_duration_median": ["mean"],
    }
    return results.groupby(groupbys).agg(aggs)


if __name__ == "__main__":
    logging.basicConfig(
        level="DEBUG",
        # format="%(asctime)s %(name)s %(levelname)s %(message)s",
        format="%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    experiments = get_started_experiments(Path("experiments"))
    experiments = filter(lambda x: x.complete(), experiments)
    # experiments = filter(lambda x: "2024-07" in str(x.path), experiments)
    experiments = filter(lambda x: x.args.model in ("mnist", "rnaseq"), experiments)
    # experiments = filter(lambda x: x.args.K == 5, experiments)
    # experiments = filter(lambda x: x.args.epochs == 256, experiments)
    experiments = filter(
        lambda x: x.args.posterior in ("Normal", "RiemannianNormal"), experiments
    )
    experiments = filter(lambda x: x.args.lr == 0.0005, experiments)
    experiments = filter(lambda x: x.args.batch_size == 128, experiments)
    # experiments = filter(lambda x: x.args.data_size == [50], experiments)
    df_losses = combine_loss_dfs(experiments)
    # starteds = get_last_losses(df_losses)
    # pprint(starteds.tail(10))
    # filter to only the last epoch row in each experiment_dir
    filters_all = (
        df_losses["is_completion_epoch"]
        # df_losses["is_last_epoch"]
        & (df_losses["batch_size"] == 128)
        & (df_losses["iwae_samples"] == 5000)
        & (df_losses["latent_dim"] == 2)
        & (df_losses["lr"] == 0.0005)
    )
    print(filters_all.mean(), filters_all.sum())
    filters_mnist = (
        (df_losses["model"] == "mnist")
        & (df_losses["K"] == 1)
        & (df_losses["epochs"] == 80)
        & (df_losses["hidden_dim"] == 600)
        # & df_losses["posterior"].isin(["Normal", "WrappedNormalDifferentLogProb"])
        # & (df_losses["seed"] == 42)
    )
    filters_rnaseq = (
        (df_losses["model"] == "rnaseq")
        & (df_losses["K"] == 5)
        & (df_losses["epochs"] == 256)
        & (df_losses["hidden_dim"] == 50)
        # & (df_losses["posterior"] in ("Normal", "RiemannianNormal"))
    )
    df_losses = df_losses[filters_all & (filters_mnist | filters_rnaseq)]
    aggregations = summarize_complete_experiments(df_losses)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(aggregations)
    for value in aggregations[("test_mlik", "mean_pom_95ci")].values:
        print(value)
    # aggregations.to_csv("aggregations.csv")
    # pprint(summarize_started_experiments(df_losses))
    # exper = ExperimentOutput.from_experiment_dir("2024-04-05T03_36_55.158096bvwyfqlm")
    # pprint(exper.make_loss_df())
    # pprint(exper.get_epoch_durations())
    print(df_losses[filters_mnist][["posterior", "c", "test_mlik", "experiment_dir"]].sort_values(["posterior", "c", "test_mlik"]))
