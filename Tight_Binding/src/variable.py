# Author : Bayu Aditya
import os
import json
import numpy as np
import argparse
from mpi4py import MPI

# DEBUG MODE
parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file JSON", type=str)
args = parser.parse_args()


class mpi_information:
    def __init__(self, communication):
        self.comm = communication
        self.size = communication.Get_size()
        self.rank = communication.Get_rank()

    @property
    def size_nodes(self):
        names = MPI.Get_processor_name()
        names = self.comm.gather(names, root=0)
        return len(np.unique(np.array(names)))
    
    def summary_mpi(self):
        print("PARALLELIZATION INFO")
        print("    Parallel version (MPI4PY) running on {} processors".format(self.size))
        print("    MPI processes distributed on {} nodes \n".format(self.size_nodes))


class jsonio:
    def __init__(self, input_json):
        self._data = self._read_json(input_json)
        self._data["num_orbitals"] = None
        self._input_json = input_json

    @property
    def hr_file(self):
        return self._data["system"]["hr_file"]

    @property
    def atomic_position(self):
        return self._data["system"]["atomic_position"]

    @property
    def orbital_index(self):
        return self._data["system"]["orbital_index"]

    @property
    def a(self):
        return self._data["cell_parameter"]["a"]

    @property
    def b(self):
        return self._data["cell_parameter"]["b"]

    @property
    def c(self):
        return self._data["cell_parameter"]["c"]

    @property
    def max_cell(self):
        return self._data["cell_parameter"]["max_cell"]

    @property
    def kpoint(self):
        return self._data["k-points"]["nk"]

    @property
    def num_k(self):
        return self.kpoint[0] + self.kpoint[1] + self.kpoint[2]

    @property
    def start(self):
        return self._data["DOS"]["start"]
        
    @property
    def stop(self):
        return self._data["DOS"]["stop"]

    @property
    def num_freq(self):
        return self._data["DOS"]["num"]
        
    @property
    def zero_plus(self):
        return self._data["DOS"]["zero_plus"]

    @property
    def step(self):
        return self._data["DOS"]["step"]

    @property
    def outdir(self):
        return self._data["output"]["loc"]

    @property
    def hamiltonian(self):
        return self._data["output"]["hamiltonian"]

    def summary_IO(self):
        print("INPUT CONFIGURATION")
        print("    input file JSON          : {}".format(self._input_json))
        print("    hr file                  : {}".format(self.hr_file))
        print("    atomic position          : {}".format(self.atomic_position))
        print("    orbital index            : {}\n".format(self.orbital_index))
        print("OUTPUT CONFIGURATION")
        print("    directory                : {}".format(self.outdir))
        print("    hamiltonian names        : {}\n".format(self.hamiltonian))

    def _read_json(self, input_json):
        with open(input_json, "r") as read:
            data = json.load(read)
        return data


class variable(jsonio, mpi_information):
    def __init__(self, communication, input_json):
        mpi_information.__init__(self, communication)
        jsonio.__init__(self, input_json)

    def generate_json(self):
        assert type(self._data["num_orbitals"]) != type(None), "num_orbitals is None"
        loc_json_output = os.path.join(self.outdir, "input4dos.json")
        with open(loc_json_output, 'w') as json_file:
            json.dump(self._data, json_file, indent=4)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    var = mpi_information(comm)
    var.summary_mpi()

    var = jsonio(args.input)
    var.summary_IO()

    var = variable(comm, args.input)
    var.generate_json()