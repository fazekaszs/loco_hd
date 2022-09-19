import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Union, Tuple
from time import time

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO, Select

from loco_hd import LoCoHD

ATOM_ID = Tuple[int, str]
TERMINAL_O = ["OT1", "OT2", "OC1", "OC2", "OXT"]
PRIMITIVE_TYPES = ["O_neg", "O_neu", "N_pos", "N_neu", "C_ali", "C_aro", "S"]
SIDECHAIN_ATOMS = [
        ["GLU:OE1", "GLU:OE2", "ASP:OD1", "ASP:OD2"],
        ["GLN:OE1", "ASN:OD1", "SER:OG", "THR:OG1", "TYR:OH"],
        ["ARG:NE", "ARG:NH1", "ARG:NH2", "LYS:NZ"],
        ["GLN:NE2", "ASN:ND2", "HIS:ND1", "HIS:NE2", "TRP:NE1"],
        ["ALA:CB", "VAL:CB", "VAL:CG1", "VAL:CG2", "ILE:CB", "ILE:CG1",
         "ILE:CG2", "ILE:CD1", "ILE:CD", "LEU:CB", "LEU:CG", "LEU:CD1",
         "LEU:CD2", "PHE:CB", "SER:CB", "THR:CB", "THR:CG2", "ASP:CB",
         "ASP:CG", "ASN:CB", "ASN:CG", "GLU:CB", "GLU:CG", "GLU:CD",
         "GLN:CB", "GLN:CG", "GLN:CD", "ARG:CB", "ARG:CG", "ARG:CD",
         "ARG:CE", "ARG:CZ", "LYS:CB", "LYS:CG", "LYS:CD", "LYS:CE",
         "HIS:CB", "CYS:CB", "MET:CB", "MET:CG", "MET:CE", "PRO:CB",
         "PRO:CG", "PRO:CD", "TYR:CB", "TRP:CB"],
        ["HIS:CG", "HIS:CD2", "HIS:CE1", "PHE:CG", "PHE:CD1", "PHE:CD2",
         "PHE:CE1", "PHE:CE2", "PHE:CZ", "TYR:CG", "TYR:CD1", "TYR:CD2",
         "TYR:CE1", "TYR:CE2", "TYR:CZ", "TRP:CG", "TRP:CD1", "TRP:CD2",
         "TRP:CE2", "TRP:CE3", "TRP:CZ2", "TRP:CZ3", "TRP:CH2"],
        ["CYS:SG", "MET:SD"]
]


class AtomSelector(Select):

    def __init__(self, accepted_atoms: List[ATOM_ID]):
        super().__init__()
        self.accepted_atoms = accepted_atoms

    def accept_atom(self, atom: Atom):

        resi: Residue = atom.parent
        if (resi.get_id()[1], atom.get_name()) in self.accepted_atoms:
            return True
        return False


def assign_primitive_type(resi_name: str, atom_name: str) -> Union[str, None]:

    # Backbone atoms:
    if atom_name in ["CA", "C"]:
        return "C_ali"
    if atom_name == "O":
        return "O_neu"
    if atom_name == "N":
        return "N_neu"
    if atom_name in TERMINAL_O:
        return "O_neg"

    # Sidechain atoms:
    for group_idx, atom_group in enumerate(SIDECHAIN_ATOMS):
        if f"{resi_name}:{atom_name}" in atom_group:
            return PRIMITIVE_TYPES[group_idx]

    return None


def assign_primitive_structure(chain: Chain) -> Tuple[List[ATOM_ID], List[str]]:

    primitive_sequence = list()
    accepted_atoms = list()

    atom: Atom
    for atom in chain.get_atoms():

        primitive_atom = assign_primitive_type(atom.parent.get_resname(), atom.get_name())
        if primitive_atom is not None:

            primitive_sequence.append(primitive_atom)

            atom_id = atom.full_id
            atom_id = (atom_id[3][1], atom_id[4][0])
            accepted_atoms.append(atom_id)

    return accepted_atoms, primitive_sequence


def plot_result(chain_chain_dmx_mean: np.ndarray,
                lchd_by_atom: np.ndarray,
                dmx_lchd_min: float,
                dmx_lchd_max: float,
                save_dir: Path,
                save_name: str):

    fig, ax = plt.subplots(1, 2)

    im = ax[0].imshow(chain_chain_dmx_mean, cmap="coolwarm", vmin=dmx_lchd_min, vmax=dmx_lchd_max)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar_ticks = np.arange(dmx_lchd_min, dmx_lchd_max, 0.025)
    cbar.set_ticks(cbar_ticks, labels=[f"{tick:.1%}" for tick in cbar_ticks])

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("LoCoHD Score of\nthe Structure Pairs")

    ax[1].plot(np.sort(lchd_by_atom), c="black")
    ax[1].set_xlabel("Rank of atom by sorted LoCoHD score")
    ax[1].set_ylabel("LoCoHD score")
    ax[1].set_title("Distribution of\nAtom LoCoHD Scores")

    atom_lchd_mean = np.mean(lchd_by_atom)
    atom_lchd_std = np.std(lchd_by_atom)
    ax[1].plot([0, len(lchd_by_atom)], [atom_lchd_mean, atom_lchd_mean], c="red")
    ax[1].fill_between([0, len(lchd_by_atom)],
                       2 * [atom_lchd_mean - atom_lchd_std, ],
                       2 * [atom_lchd_mean + atom_lchd_std, ],
                       alpha=0.25, color="red", edgecolor=None)

    atom_lchd_min = np.min(lchd_by_atom)
    atom_lchd_max = np.max(lchd_by_atom)
    plot_ticks = np.arange(atom_lchd_min, atom_lchd_max, (atom_lchd_max - atom_lchd_min) / 10)
    ax[1].set_yticks(plot_ticks, labels=[f"{tick:.1%}" for tick in plot_ticks])

    atom_lchd_median = np.median(lchd_by_atom)
    legend_labels = list()
    legend_labels.append(f"Min = {atom_lchd_min:.1%}")
    legend_labels.append(f"Max = {atom_lchd_max:.1%}")
    legend_labels.append(f"Mean = {atom_lchd_mean:.1%}")
    legend_labels.append(f"Median = {atom_lchd_median:.1%}")
    legend_labels.append(f"StD = {atom_lchd_std:.1%}")
    legend_handles = Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)
    legend_handles = [legend_handles, ] * len(legend_labels)
    ax[1].legend(legend_handles, legend_labels,
                 loc="best", fontsize="small", fancybox=True,
                 framealpha=0.7, handlelength=0, handletextpad=0)

    plt.tight_layout()
    fig.savefig(save_dir / f"{save_name}_plot.png", dpi=300)


def compare_structures(prot_root_path: Path, save_dir: Path, save_name: str):

    print(f"Starting {save_name}...")

    # Read proteins
    prot_chains: List[Chain] = list()
    all_files = os.listdir(prot_root_path)  # [:10]
    file_name: str
    for file_name in all_files:
        prot_structure = PDBParser(QUIET=True).get_structure("", prot_root_path / file_name)
        prot_chains.append(prot_structure[0].child_list[0])
    del prot_structure

    # Collect primitive atom types
    atom: Atom
    accepted_atoms, primitive_sequence = assign_primitive_structure(prot_chains[0])
    print(f"Number of primitive atoms: {len(primitive_sequence)}")

    # Collect atom distance matrices
    dmx_collection = list()
    for chain in prot_chains:

        dmx = list()
        for atom_id in accepted_atoms:
            coord = chain[atom_id[0]][atom_id[1]].coord
            dmx.append(coord)

        dmx = np.array(dmx)
        dmx = dmx[np.newaxis, :, :] - dmx[:, np.newaxis, :]
        dmx = np.sqrt(np.sum(dmx ** 2, axis=2))
        dmx_collection.append(dmx)

    # Initialize LoCoHD instance
    lchd = LoCoHD(PRIMITIVE_TYPES, ("dagum", [13.4, 6.4, 16.2]))

    # Calculate lchd mean and std values
    runtimes = list()
    chain_chain_dmx_mean = np.zeros((len(prot_chains), len(prot_chains)))
    lchd_by_atom = list()
    for idx1 in range(len(prot_chains)):
        for idx2 in range(idx1 + 1, len(prot_chains)):

            start_time = time()
            lchd_all = lchd.from_dmxs(primitive_sequence,
                                      primitive_sequence,
                                      dmx_collection[idx1],
                                      dmx_collection[idx2])
            end_time = time()
            runtimes.append(end_time - start_time)
            lchd_by_atom.append(lchd_all)

            chain_chain_dmx_mean[idx1, idx2] = chain_chain_dmx_mean[idx2, idx1] = np.mean(lchd_all)

    lchd_by_atom = np.mean(lchd_by_atom, axis=0)
    print(f"Mean time per run: {np.mean(runtimes):.5f} s")
    print(f"Std of runtimes: {np.std(runtimes):.10f} s")
    print(f"Total runtime: {np.sum(runtimes):.5f} s")

    sorted_dmx_mask = np.argsort(np.mean(chain_chain_dmx_mean, axis=0))
    chain_chain_dmx_mean = chain_chain_dmx_mean[:, sorted_dmx_mask][sorted_dmx_mask, :]

    # Save b-factor labelled structure
    pdb_io = PDBIO()

    for lchd_score, atom_id in zip(lchd_by_atom, accepted_atoms):

        prot_chains[0][atom_id[0]][atom_id[1]].bfactor = 100 * lchd_score
            
    pdb_io.set_structure(prot_chains[0])
    pdb_io.save(str(save_dir / f"{save_name}_blabelled.pdb"), select=AtomSelector(accepted_atoms))

    return chain_chain_dmx_mean, lchd_by_atom


def main():

    save_dir = Path("./workdir/prot_batch_resuls")
    paths_and_names = [
        # ("./workdir/pdb_files/h5", "h5_dummy"),
        ("/home/fazekaszs/CoreDir/PhD/PDB/H5/277", "h5_277"),
        ("/home/fazekaszs/CoreDir/PhD/PDB/H5/288", "h5_288"),
        ("/home/fazekaszs/CoreDir/PhD/PDB/H5/299", "h5_299"),
        ("/home/fazekaszs/CoreDir/PhD/PDB/H5/310", "h5_310"),
        ("/home/fazekaszs/CoreDir/PhD/PDB/H5/321", "h5_321"),
    ]

    all_chain_chain_dmxs, all_lchd_by_atom = list(), list()

    for prot_root_path, prot_name in paths_and_names:

        temp_dmx, temp_atom_lchds = compare_structures(Path(prot_root_path), save_dir, prot_name)
        all_chain_chain_dmxs.append(temp_dmx)
        all_lchd_by_atom.append(temp_atom_lchds)

    print("Starting to plot!")
    dmx_lchd_min = np.min(all_chain_chain_dmxs)
    dmx_lchd_max = np.max(all_chain_chain_dmxs)

    for chain_chain_dmx, lchd_by_atom, (_, prot_name) in zip(all_chain_chain_dmxs, all_lchd_by_atom, paths_and_names):
        plot_result(chain_chain_dmx, lchd_by_atom, dmx_lchd_min, dmx_lchd_max, save_dir, prot_name)


if __name__ == "__main__":
    main()
