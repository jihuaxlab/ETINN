# usage
# python ETINN_eval.py path_to_your_model --data path_to_xyz_files --labels path_to_labels

# lumo
# python ETINN_eval.py ./models/ETINN_cutoff_4_4_lumo/model_output/  --labels ./data/lumo-old --data ./data/xyz-old-fix/  --output_dir ./log/test-lumo

# homo
# python ETINN_eval.py ./models/ETINN_cutoff_4_4/model_output/  --labels ./data/homo-old --data ./data/xyz-old-fix/  --output_dir ./log/test

import logging
import os
import json
import argparse
import math
import contextlib
import timeit

import ase
import ase.io
import torch
import numpy as np

# import my_dataset as dataset
import orb_dataset as dataset
import attnmodel_vbi_produit as attnmodel

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Predict with pretrained model", fromfile_prefix_chars="@"
    )
    parser.add_argument("model_dir", type=str, help='Directory of pretrained model')
    parser.add_argument("--output_dir", type=str, default=r"./log", help="Output directory")
    parser.add_argument("--probe_count", type=int, default=100, help="probe_count is equal to atoms count")
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json in the training process.",
    )
    parser.add_argument(
        "--data", type=str, default=r"./data/xyz", help="Path to ASE database",
    )
    parser.add_argument(
        "--labels", type=str, default=None , help="Path to orb labels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Set which device to use for inference e.g. 'cuda' or 'cpu'",
    )
    

    return parser.parse_args(arg_list)

def load_model(model_dir, device, args=None):
    with open(os.path.join(model_dir, "arguments.json"), "r") as f:
        runner_args = argparse.Namespace(**json.load(f))
        print(runner_args)
        if args is not None:
            args.split_file = runner_args.split_file
    if runner_args.use_painn_model:
        model = attnmodel.PainnETINNModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff, cutoff_prob=runner_args.cutoff_prob)
    else:
        model = attnmodel.ETINNModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff, cutoff_prob=runner_args.cutoff_prob)
    device = torch.device(device)
    model.to(device)
    state_dict = torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state_dict["model"])
    return model, runner_args.cutoff

def split_data(dataset, args):
    # Load or generate splits
    if os.path.isfile(args.split_file):
        print("load split file:", args.split_file)
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        raise Exception("should give the split data file to launch evaluate process.")

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits
    
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

    model, cutoff = load_model(args.model_dir, args.device, args)

    
    densitydata = dataset.OrbitalData(args.data, args.labels)
    
    # densitydata = torch.utils.data.ConcatDataset([dataset.DensityData(path) for path in filelist])

    # Split data into train and validation sets
    datasplits = split_data(densitydata, args)
    val_dataset = datasplits["validation"]
    
    mse_list = []
    for i in range(len(val_dataset)):

        density_dict = val_dataset[i]
        num_atoms = len(density_dict["atoms"])
        print("loaded file {}, contains {} atoms".format(density_dict["metadata"], num_atoms))

        # print(density_dict)
        

        device = torch.device(args.device)

        # ase
        # atoms.set_pbc(True)	晶体、块体材料
        # atoms.set_pbc([True, True, False])	二维材料、表面
        # atoms.set_pbc(False)	分子、团簇、非周期系统
        pbc_info = False

        start_time = timeit.default_timer()

        with torch.no_grad():
            # Make graph with no probes
            logging.debug("Computing atom-to-atom graph")
            collate_fn = dataset.CollateFuncAtoms(
                cutoff=cutoff,
                pin_memory=device.type == "cuda",
                set_pbc_to=pbc_info,
            )
            graph_dict = collate_fn([density_dict])
            logging.debug("Computing atom representation")
            device_batch = {
                k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
            }
            if isinstance(model, attnmodel.PainnETINNModel):
                atom_representation_scalar, atom_representation_vector = model.atom_model(device_batch)
            else:
                atom_representation = model.atom_model(device_batch)
            logging.debug("Atom representation done")

            # Loop over all slices
            density_iter = dataset.DensityGridIterator(density_dict, probe_count=num_atoms, cutoff=cutoff, set_pbc_to=pbc_info)
            orb_distribution = []
            for probe_graph_dict in density_iter:
                probe_dict = dataset.collate_list_of_dicts([probe_graph_dict])
                probe_dict = {
                    k: v.to(device=device, non_blocking=True) for k, v in probe_dict.items()
                }
                device_batch["probe_edges"] = probe_dict["probe_edges"]
                device_batch["probe_edges_displacement"] = probe_dict["probe_edges_displacement"]
                device_batch["probe_xyz"] = probe_dict["probe_xyz"]
                device_batch["num_probe_edges"] = probe_dict["num_probe_edges"]
                device_batch["num_probes"] = probe_dict["num_probes"]

                if isinstance(model, attnmodel.PainnETINNModel):
                    orb_distribution = model.probe_model(device_batch, atom_representation_scalar, atom_representation_vector)
                else:
                    orb_distribution = model.probe_model(device_batch, atom_representation)

                orb_distribution = orb_distribution.flatten().tolist()
                print(orb_distribution, len(orb_distribution))
                percentages_1 = np.array(orb_distribution)
                normalized_data = percentages_1 / percentages_1.sum() * 100
                # MSE
                mse = np.mean((np.array(density_dict["density"]) - normalized_data) ** 2)
                print("MSE: ", mse)
                mse_list.append(mse)
                json_data = json.dumps({"prediction": normalized_data.tolist(), "label":density_dict["density"], "mse": mse})
                log_dir = args.output_dir
                os.makedirs(log_dir, exist_ok=True)
                if os.path.isdir(args.output_dir):
                    log_dir = args.output_dir
                
                base_name_with_extension = density_dict["metadata"]['filename']
                base_name, extension = os.path.splitext(base_name_with_extension)
                out_json_path = os.path.join(log_dir, base_name + "_with_label.txt")
                with open(out_json_path, 'w') as f:
                    f.write(json_data)
                print(f"Result written in :{out_json_path}")

        end_time = timeit.default_timer()
        logging.info("done time_elapsed=%f", end_time - start_time)
    print("Final MSE: ", np.mean(mse_list))

if __name__ == "__main__":
    main()
