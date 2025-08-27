# usage
# Training: run python ETINN_main.py --data path_to_xyz_files --labels path_to_labels --cutoff 4 --cutoff_prob 4 --num_interactions 4 --split_file ./data/datasplits.json --output_dir  path_to_output
# example:
# python ETINN_main.py --data ./data/xyz-old-fix --labels ./data/homo-old --cutoff 4 --cutoff_prob 4 --num_interactions 4 --max_steps 50000000 --node_size 128 --split_file ./data/datasplits.json --output_dir ./logs/runs_homo_old_xyz_fix_4-4_produit_output_l4/model_output --device cuda:0

import os
import sys
import json
import argparse
import math
import logging
import itertools
import timeit

import numpy as np
import torch
import torch.utils.data

torch.set_num_threads(1)  # Try to avoid thread overload on cluster

import attnmodel_vbi_produit as attnmodel
import orb_dataset as my_dataset


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        help="Atomic interaction cutoff distance [Å]",
    )
    parser.add_argument(
        "--cutoff_prob",
        type=float,
        default=4.0,
        help="Atomic interaction cutoff distance [Å]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--data", type=str, default=r"/mnt/LargeStorageSpace/HEZhaoming/DeepDFT/data/xyz", help="Path to ASE database",
    )
    parser.add_argument(
        "--labels", type=str, default=r"/mnt/LargeStorageSpace/HEZhaoming/DeepDFT/data/homo",
        help="Path to ASE database",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(1e6),
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )

    parser.add_argument(
        "--use_painn_model",
        action="store_true",
        help="Enable equivariant message passing model (PaiNN)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of training",
    )

    return parser.parse_args(arg_list)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.05))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

        # Save split file
        with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
            json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits


def eval_model(model, dataloader, device):
    with torch.no_grad():
        running_ae = torch.tensor(0., device=device)
        running_se = torch.tensor(0., device=device)
        running_count = torch.tensor(0., device=device)
        for batch in dataloader:
            device_batch = {
                k: v.to(device=device, non_blocking=True) for k, v in batch.items()
            }
            outputs = model(device_batch)
            targets = device_batch["probe_target"]

            running_ae += torch.sum(torch.abs(targets - outputs))
            running_se += torch.sum(torch.square(targets - outputs))
            running_count += torch.sum(device_batch["num_probes"])

        mae = (running_ae / running_count).item()
        rmse = (torch.sqrt(running_se / running_count)).item()

    return mae, rmse


def get_normalization(dataset, per_atom=True):
    try:
        num_targets = len(dataset.transformer.targets)
    except AttributeError:
        num_targets = 1
    x_sum = torch.zeros(num_targets)
    x_2 = torch.zeros(num_targets)
    num_objects = 0
    for sample in dataset:
        x = sample["targets"]
        if per_atom:
            x = x / sample["num_nodes"]
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0

    return x_mean, torch.sqrt(x_var)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # m_data_dir = r"/mnt/LargeStorageSpace/HEZhaoming/DeepDFT/data/xyz"
    # m_label_dir = r"/mnt/LargeStorageSpace/HEZhaoming/DeepDFT/data/homo"

    # logging.info("loading data %s", args.dataset)
    logging.info("loading data %s", logging.info("loading data %s", args.data))
    logging.info("loading labels %s", logging.info("loading data %s", args.labels))

    # 创建数据集实例
    orbital_data = my_dataset.OrbitalData(args.data, args.labels)

    # densitydata = torch.utils.data.ConcatDataset([dataset.DensityData(path) for path in filelist])

    # Split data into train and validation sets
    datasplits = split_data(orbital_data, args)
    datasplits["train"] = my_dataset.RotatingPoolData(datasplits["train"], 20)

    # ase
    # atoms.set_pbc(True)	晶体、块体材料
    # atoms.set_pbc([True, True, False])	二维材料、表面
    # atoms.set_pbc(False)	分子、团簇、非周期系统
    pbc_info = False

    # Setup loaders
    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        args.batch_size,
        num_workers=4,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=my_dataset.CollateFuncRandomSample(args.cutoff, 300, pin_memory=False, set_pbc_to=pbc_info),
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"],
        args.batch_size,
        collate_fn=my_dataset.CollateFuncRandomSample(args.cutoff, 300, pin_memory=False, set_pbc_to=pbc_info),
        num_workers=0,
    )
    logging.info("Preloading validation batch")
    val_loader = [b for b in val_loader]

    # Initialise model
    device = torch.device(args.device)
    if args.use_painn_model:
        net = attnmodel.PainnModel(args.num_interactions, args.node_size, args.cutoff, cutoff_prob=args.cutoff_prob)
    else:
        net = attnmodel.ETINNModel(args.num_interactions, args.node_size, args.cutoff, cutoff_prob=args.cutoff_prob)

    logging.debug("model has %d parameters", count_parameters(net))
    net = net.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

    log_interval = 5000
    running_loss = torch.tensor(0.0, device=device)
    running_loss_count = torch.tensor(0, device=device)
    best_val_mae = np.inf
    step = 0
    # Restore checkpoint
    if args.load_model:
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        step = state_dict["step"]
        best_val_mae = state_dict["best_val_mae"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])

    logging.info("start training")

    data_timer = AverageMeter("data_timer")
    transfer_timer = AverageMeter("transfer_timer")
    train_timer = AverageMeter("train_timer")
    eval_timer = AverageMeter("eval_time")

    endtime = timeit.default_timer()
    for _ in itertools.count():
        for batch_host in train_loader:
            data_timer.update(timeit.default_timer() - endtime)
            tstart = timeit.default_timer()
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            transfer_timer.update(timeit.default_timer() - tstart)

            tstart = timeit.default_timer()
            # Reset gradient
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = net(batch)
            loss = criterion(outputs, batch["probe_target"])
            # print(outputs.shape, batch["probe_target"].shape) # => [1, n_atom]
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss * batch["probe_target"].shape[0] * batch["probe_target"].shape[1]
                running_loss_count += torch.sum(batch["num_probes"])

            train_timer.update(timeit.default_timer() - tstart)

            # print(step, loss_value)
            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                tstart = timeit.default_timer()
                with torch.no_grad():
                    train_loss = (running_loss / running_loss_count).item()
                    running_loss = running_loss_count = 0

                val_mae, val_rmse = eval_model(net, val_loader, device)

                logging.info(
                    "step=%d, val_mae=%g, val_rmse=%g, sqrt(train_loss)=%g",
                    step,
                    val_mae,
                    val_rmse,
                    math.sqrt(train_loss),
                )

                # Save checkpoint
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_mae": best_val_mae,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )

                eval_timer.update(timeit.default_timer() - tstart)
                logging.debug(
                    "%s %s %s %s" % (data_timer, transfer_timer, train_timer, eval_timer)
                )
            step += 1

            scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                sys.exit(0)

            endtime = timeit.default_timer()


if __name__ == "__main__":
    main()
