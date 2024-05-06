import json
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import pandas as pd
import torch


class ExperimentOutput:
    def __init__(self, path: Path):
        self.path = path
        self.args = torch.load(self.path / "args.rar")  # returns argparse.Namespace
        self.args_json = json.load(open(self.path / "args.json"))
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
        if not (self.path / "recon_{:03d}.png".format(experiment_epochs)).exists():
            return False
        return True

    def get_epoch_durations(self) -> Dict[int, float]:
        # start = timestamp of creation of self.path / "args.rar"
        # end of each epoch = timestamp of "gen_means_{:03d}.png".format(epoch)
        # end of experiment = timestamp of "run.log"
        start_of_first_epoch = (self.path / "args.rar").stat().st_ctime
        # fpr each "gen_means_*" file, get epoch number and timestamp
        end_times = {
            int(p.name.split("_")[2][:3]): p.stat().st_ctime
            for p in self.path.glob("gen_means_*.png")
        }
        end_times[0] = start_of_first_epoch
        durations = {i: end - end_times[i - 1] for i, end in end_times.items() if i > 0}
        return durations

    def _load_losses_rar(self) -> Dict[str, list]:
        # keep this method for type annotation
        return torch.load(self.path / "losses.rar")

    def get_loss_df(self) -> pd.DataFrame:
        losses = self._load_losses_rar()
        args_to_include = {
            k: v for k, v in vars(self.args).items() if not isinstance(v, (list, dict))
        }
        df = pd.DataFrame(
            {
                "experiment_dir": self.path.name,
                "started": self.started(),
                "stopped": self.stopped(),
                "complete": self.complete(),
                "epoch": range(1, len(losses["train_loss"]) + 1),
                "test_loss": losses["test_loss"],
                "test_mlik": losses["test_mlik"],
                "train_loss": losses["train_loss"],
                "train_recon": losses["train_recon"],
                "train_kl": losses["train_kl"],
                **args_to_include,
            }
        )
        for k in losses.keys():
            # replace extreme values with NaN
            df[k] = df[k].apply(lambda x: x if 0 < x < 1e4 else None)
        df["is_completion_epoch"] = df["epoch"] == df["epochs"]
        df["is_last_epoch"] = df["epoch"] == df["epoch"].max()
        epoch_durations = self.get_epoch_durations()
        df["epoch_duration"] = df["epoch"].map(epoch_durations)
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
        except FileNotFoundError:
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
        "started",
        "stopped",
        "complete",
        "epoch",
        "train_loss",
        "test_loss",
        "epoch_duration_median",
    ]
    return df[cols]


def summarize_complete_experiments(results: pd.DataFrame) -> pd.DataFrame:
    results = results[results["is_completion_epoch"]]
    groupbys = [
        "manifold",
        "posterior",
        "c",
        "dec",
        "epochs",
    ]

    def std_pop(x):
        return x.std(ddof=0)

    def mean_pom_95ci(x):
        mean = x.mean()
        ci = 1.96 * x.std()
        n = x.count()
        return "{:.1f}±{:.1f} (n={})".format(mean, ci, n)

    aggs = {
        # "test_loss": ["count", "mean", "std"],
        "test_mlik": [mean_pom_95ci, "median"],
        # "train_loss": ["count", "mean", "std"],
        # "train_recon": ["count", "mean", "std", mean_pom_95ci],
        # "train_kl": ["count", "mean", "std"],
        "epoch_duration_median": ["mean"],
    }
    return results.groupby(groupbys).agg(aggs)


def summarize_started_experiments(results: pd.DataFrame) -> pd.DataFrame:
    groupbys = [
        "manifold",
        "posterior",
        "c",
        "dec",
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
    experiments = get_started_experiments(Path("experiments"))
    df_losses = combine_loss_dfs(experiments)
    starteds = get_last_losses(df_losses)
    pprint(starteds)
    pprint(summarize_complete_experiments(df_losses))
    pprint(summarize_started_experiments(df_losses))
    # exper = ExperimentOutput.from_experiment_dir("2024-04-05T03_36_55.158096bvwyfqlm")
    # pprint(exper.make_loss_df())
    # pprint(exper.get_epoch_durations())
