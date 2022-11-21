import json
from pathlib import Path

import click
import numpy as np
import torch

from util import data
from util.trainer import get_trainer


@click.command()
@click.option(
    "--method", "-m", default="dann", type=click.Choice(["dann", "source", "target"])
)
@click.option(
    "--out-dir",
    "-o",
    default="results/default",
    type=click.Path(exists=False, writable=True),
)
@click.option("--debug", is_flag=True)
def main(method: str, out_dir: str, debug: bool) -> None:

    dataset_train, dataset_test = data.load(method)
    D = dataset_train.X.shape[1]
    K = int(torch.cat((dataset_train.y, dataset_test.y)).max()) + 1
    n_epochs = 2 if debug else 500
    train_args = {"n_epochs": n_epochs, "device": "cuda:0", "batch_size": 1024}
    model_args = {
        "D": D,
        "K": K,
        "d_hidden": 1000,
        "d_latent": 1000,
        "n_layers": 1,
        "scale": 0.1,
    }
    trainer = get_trainer(model_args, train_args)

    np.set_printoptions(linewidth=200)
    trainer.train(dataset_train, dataset_test)
    result = trainer.result

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    with open(out_dir_path / "result.json", "w") as f:
        json.dump(result, f)

    args = {"method": method, "out_dir": out_dir}
    with open(out_dir_path / "args.json", "w") as f:
        json.dump(args, f)


if __name__ == "__main__":
    main()
