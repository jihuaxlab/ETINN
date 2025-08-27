
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import multiprocessing
import queue
import time
import threading
import logging
import os
import torch
import torch.utils.data
import numpy as np
import ase
import ase.neighborlist
import ase.io.cube
import ase.units
import asap3

from layer import pad_and_stack


def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights


def rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)).tolist():
            queue.put(dataset[index])


def transfer_thread(queue: multiprocessing.Queue, datalist: list):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()


class RotatingPoolData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset that continously loads data into a smaller pool.
    The data loading is performed in a separate process and is assumed to be IO bound.
    """

    def __init__(self, dataset, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        self.data_pool = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            ).tolist()
        ]
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
        )
        self.transfer_thread = threading.Thread(
            target=transfer_thread, args=(self.loader_queue, self.data_pool)
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index):
        return self.data_pool[index]


class BufferData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset. Loads all data into memory.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)

        self.data_objects = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]


class OrbitalData(torch.utils.data.Dataset):
    def __init__(self, datapath, label_dir, **kwargs):
        super().__init__(**kwargs)
        if os.path.isfile(datapath) and datapath.endswith(".tar"):
            raise ValueError("Tar dataset at path %s", datapath)
            self.data = DensityDataTar(datapath)
        elif os.path.isdir(datapath):
            self.data = OrbitalDataDir(datapath, label_dir)
        else:
            raise ValueError("Did not find dataset at path %s", datapath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class OrbitalDataDir(torch.utils.data.Dataset):
    def __init__(self, directory, label_dir, **kwargs):
        super().__init__(**kwargs)

        self.directory = directory
        self.member_list = sorted(os.listdir(self.directory))

        self.label_dir = label_dir
        self.key_to_idx = {str(k): i for i, k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def extractfile(self, filename):
        path = os.path.join(self.directory, filename)
        if path.endswith(".xyz"):
            atoms = _read_xyz(path)
        else:
            raise Exception("path not end with .xyz")

        label_path = os.path.join(self.label_dir, filename) + ".orbcomp"
        density = _read_density(label_path)
        metadata = {"filename": filename}
        return {
            "density": density,
            "atoms": atoms,
            "origin": np.zeros(3),
            "grid_position": None,
            "metadata": metadata,  # Meta information
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        return self.extractfile(self.member_list[index])


class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
                cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
                self.atoms_positions[indices]
                + offsets @ self.atoms_cell
                - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


def grid_iterator_worker(atoms, meshgrid, probe_count, cutoff, slice_id_queue, result_queue):
    try:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)
    except Exception as e:
        logging.info("Failed to create asap3 neighborlist, this might be very slow. Error: %s", e)
        neighborlist = None
    while True:
        try:
            slice_id = slice_id_queue.get(True, 1)
        except queue.Empty:
            while not result_queue.empty():
                time.sleep(1)
            result_queue.close()
            return 0
        res = DensityGridIterator.static_get_slice(slice_id, atoms, meshgrid, probe_count, cutoff,
                                                   neighborlist=neighborlist)
        result_queue.put((slice_id, res))


class DensityGridIterator:
    def __init__(self, densitydict, probe_count: int, cutoff: float, set_pbc_to: Optional[bool] = None):
        # num_positions = np.prod(densitydict["grid_position"].shape[0:3])
        self.num_slices = 1
        self.probe_count = probe_count
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

        if self.set_pbc is not None:
            self.atoms = densitydict["atoms"].copy()
            self.atoms.set_pbc(self.set_pbc)
        else:
            self.atoms = densitydict["atoms"]

        self.meshgrid = densitydict["grid_position"]

    def get_slice(self, slice_index):
        return self.static_get_slice(slice_index, self.atoms, self.meshgrid, self.probe_count, self.cutoff)

    @staticmethod
    def static_get_slice(slice_index, atoms, meshgrid, probe_count, cutoff, neighborlist=None):
        atoms_pos = atoms.get_positions()
        # print(atoms_pos)

        # 原子数小于探针总数则循环添加直至到达探针总数
        probe_pos = []
        while len(probe_pos) < probe_count:
            probe_pos.extend(atoms_pos)
        probe_pos = probe_pos[:probe_count]
        probe_pos = np.array(probe_pos)

        atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = atoms_to_graph(atoms, cutoff)

        probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist=neighborlist,
                                                                inv_cell_T=inv_cell_T)

        if not probe_edges:
            probe_edges = [np.zeros((0, 2), dtype=np.int)]
            probe_edges_displacement = [np.zeros((0, 3), dtype=np.float32)]

        res = {
            "probe_edges": np.concatenate(probe_edges, axis=0),
            "probe_edges_displacement": np.concatenate(probe_edges_displacement, axis=0).astype(np.float32),
        }
        res["num_probe_edges"] = res["probe_edges"].shape[0]
        res["num_probes"] = probe_count
        res["probe_xyz"] = probe_pos.astype(np.float32)

        return res

    def __iter__(self):
        self.current_slice = 0
        slice_id_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue(100)
        self.finished_slices = dict()
        for i in range(self.num_slices):
            slice_id_queue.put(i)
        self.workers = [multiprocessing.Process(target=grid_iterator_worker, args=(
        self.atoms, self.meshgrid, self.probe_count, self.cutoff, slice_id_queue, self.result_queue)) for _ in range(6)]
        for w in self.workers:
            w.start()
        return self

    def __next__(self):
        if self.current_slice < self.num_slices:
            this_slice = self.current_slice
            self.current_slice += 1

            # Retrieve finished slices until we get the one we are looking for
            while this_slice not in self.finished_slices:
                i, res = self.result_queue.get()
                res = {k: torch.tensor(v) for k, v in res.items()}  # convert to torch tensor
                self.finished_slices[i] = res
            return self.finished_slices.pop(this_slice)
        else:
            for w in self.workers:
                w.join()
            raise StopIteration


# 添加标签和探头
def atoms_and_probe_sample_to_graph_dict(density, atoms, grid_pos, cutoff, num_probes):
    # Sample probes on the calculated grid
    atoms_pos = atoms.get_positions()
    #
    probe_pos = np.array(atoms_pos)

    probe_target = density

    atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = atoms_to_graph(atoms, cutoff)
    probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist=neighborlist,
                                                            inv_cell_T=inv_cell_T)

    default_type = torch.get_default_dtype()

    if not probe_edges:
        probe_edges = [np.zeros((0, 2), dtype=np.int)]
        probe_edges_displacement = [np.zeros((0, 3), dtype=np.int)]
    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0)),
        "probe_edges_displacement": torch.tensor(
            np.concatenate(probe_edges_displacement, axis=0), dtype=default_type
        ),
        "probe_target": torch.tensor(probe_target, dtype=default_type),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
    res["num_probes"] = torch.tensor(res["probe_target"].shape[0])
    res["probe_xyz"] = torch.tensor(probe_pos, dtype=default_type)
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res


def atoms_to_graph_dict(atoms, cutoff):
    atom_edges, atom_edges_displacement, _, _ = atoms_to_graph(atoms, cutoff)

    default_type = torch.get_default_dtype()

    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)

    return res


def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    # Compute neighborlist
    if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
            np.any(atoms.get_pbc())
            and np.any(_cell_heights(atoms.get_cell()) < cutoff)
    )
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)  # 将中心和邻接组成元组 shape = [num_neighbor, 2]

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T


def probes_to_graph(atoms, probe_pos, cutoff, neighborlist=None, inv_cell_T=None):
    probe_edges = []
    probe_edges_displacement = []
    if inv_cell_T is None:
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    if hasattr(neighborlist, "get_neighbors_querypoint"):
        results = neighborlist.get_neighbors_querypoint(probe_pos, cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
    else:
        # Insert probe atoms
        num_probes = probe_pos.shape[0]
        probe_atoms = ase.Atoms(numbers=[0] * num_probes, positions=probe_pos)
        atoms_with_probes = atoms.copy()
        atoms_with_probes.extend(probe_atoms)
        atomic_numbers = atoms_with_probes.get_atomic_numbers()

        # from ase import atoms
        # from ase.data import get_atomic_numbers
        #
        # # 获取单个原子的原子序数
        # print("Atomic number of Hydrogen:", get_atomic_numbers('H'))  # 输出: 1
        #
        # # 获取多个原子的原子序数
        # atomic_symbols = ['C', 'O', 'Fe']
        # atomic_numbers = [get_atomic_numbers(symbol) for symbol in atomic_symbols]
        # print("Atomic numbers:", atomic_numbers)  # 输出: [6, 8, 26]
        #
        # # 或者直接传入列表给get_atomic_numbers
        # print("Atomic numbers directly from list:", get_atomic_numbers(atomic_symbols))  # 同样输出: [6, 8, 26]
        #
        # # 使用ASE的Atoms对象演示
        # water = atoms('H2O')
        # numbers = [get_atomic_numbers(atom.symbol) for atom in water]
        # print("Atomic numbers in H2O:", numbers)  # 输出: [1, 1, 8]

        if (
                np.any(atoms.get_cell().lengths() <= 0.0001)
                or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        )
        ):
            neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
        else:
            neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)

        results = [neighborlist.get_neighbors(i + len(atoms), cutoff) for i in range(num_probes)]

    atom_positions = atoms.get_positions()
    for i, (neigh_idx, neigh_vec, _) in enumerate(results):
        neigh_atomic_species = atomic_numbers[neigh_idx]

        neigh_is_atom = neigh_atomic_species != 0
        neigh_atoms = neigh_idx[neigh_is_atom]
        self_index = np.ones_like(neigh_atoms) * i
        edges = np.stack((neigh_atoms, self_index), axis=1)

        neigh_pos = atom_positions[neigh_atoms]
        this_pos = probe_pos[i]
        neigh_origin = neigh_vec[neigh_is_atom] + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        probe_edges.append(edges)
        probe_edges_displacement.append(neigh_origin_scaled)

    return probe_edges, probe_edges_displacement


def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated


class CollateFuncRandomSample:
    def __init__(self, cutoff, num_probes, pin_memory=True, set_pbc_to=None):
        self.num_probes = num_probes
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        graphs = []
        mol_names = []
        for i in input_dicts:
            mol_names.append(i['metadata'])
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_and_probe_sample_to_graph_dict(
                i["density"],
                atoms,
                i["grid_position"],
                self.cutoff,
                self.num_probes,
            ))
        # print("mol_names:", mol_names)
        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)


class CollateFuncAtoms:
    def __init__(self, cutoff, pin_memory=True, set_pbc_to=None):
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                # atoms.set_pbc(True)	晶体、块体材料
                # atoms.set_pbc([True, True, False])	二维材料、表面
                # atoms.set_pbc(False)	分子、团簇、非周期系统
                atoms.set_pbc(False)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_to_graph_dict(
                atoms,
                self.cutoff,
            ))

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)


def _calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def _read_xyz(xyz_file_path):
    atoms = ase.io.read(xyz_file_path)
    return atoms


def _read_density(density_file_path):
    if os.path.isfile(density_file_path):
        with open(density_file_path, 'r') as f:
            all_data = f.readlines()
            all_data = [np.float(ele.split()[1]) for ele in all_data]
    else:
        raise Exception("File not exist {}".format(density_file_path))
    return all_data


if __name__ == "__main__":
    m_data_dir = r"./data/examples/xyz"
    m_label_dir = r"./data/examples/labels"

    # create dataset
    dataset = OrbitalData(m_data_dir, m_label_dir)

    # create data loader
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=torch.utils.data.RandomSampler(dataset),
        collate_fn=CollateFuncRandomSample(4, 1000, pin_memory=False, set_pbc_to=False),
    )

    # test batch
    for batch in dataloader:
        print(batch)