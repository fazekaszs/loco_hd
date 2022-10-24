import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Tuple
from time import time

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO, Select

from loco_hd import LoCoHD
from atom_converter_utils import PrimitiveAssigner, PrimitiveAtomTemplate

ATOM_ID = Tuple[int, str]


class AtomSelector(Select):

    def __init__(self, accepted_atoms: List[ATOM_ID]):
        super().__init__()
        self.accepted_atoms = accepted_atoms

    def accept_atom(self, atom: Atom):

        resi: Residue = atom.parent
        if (resi.get_id()[1], atom.get_name()) in self.accepted_atoms:
            return True
        return False


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

    # Create the PrimitiveAssigner instance
    primitive_assigner = PrimitiveAssigner(Path("primitive_typings/coarse_grained.config.json"))

    # Initialize LoCoHD instance
    lchd = LoCoHD(primitive_assigner.all_primitive_types, ("uniform", [0., 10.]))

    # Collect all the filenames in the directory
    all_files: List[str] = os.listdir(prot_root_path)
    all_files = list(filter(lambda x: x.endswith(".pdb"), all_files))

    # Read proteins and create the primitive atom template lists
    template_lists: List[List[PrimitiveAtomTemplate]] = list()
    file_name: str
    for file_name in all_files:

        prot_structure = PDBParser(QUIET=True).get_structure("", prot_root_path / file_name)
        primitive_atom_templates = primitive_assigner.assign_primitive_structure(prot_structure)
        template_lists.append(primitive_atom_templates)

    del file_name, prot_structure, primitive_atom_templates

    # Collect the accepted atoms and the primitive_sequence
    primitive_sequence, accepted_atoms = list(), list()
    for pra_template in template_lists[0]:

        resi_number = pra_template.atom_source.source_residue[3][1]
        atom_name = pra_template.atom_source.source_atom[0]
        accepted_atoms.append((resi_number, atom_name))

        primitive_sequence.append(pra_template.primitive_type)

    del pra_template, resi_number, atom_name

    # Create the distance matrices
    dmx_list: List[np.ndarray] = list()
    for template_list in template_lists:

        dmx = list()
        for pra_template in template_list:
            dmx.append(pra_template.coordinates)
        dmx = np.array(dmx)
        dmx = dmx[np.newaxis, ...] - dmx[:, np.newaxis, :]
        dmx = np.sqrt(np.sum(dmx ** 2, axis=2))
        dmx_list.append(dmx)

    del template_list, pra_template, dmx

    # Calculate lchd mean and std values
    n_of_structures = len(dmx_list)
    chain_chain_mean_locohd_mx = np.zeros((n_of_structures, n_of_structures))
    runtimes = list()
    lchd_by_atom = list()
    for idx1 in range(n_of_structures):
        for idx2 in range(idx1 + 1, n_of_structures):

            start_time = time()
            lchd_all = lchd.from_dmxs(primitive_sequence,
                                      primitive_sequence,
                                      dmx_list[idx1],
                                      dmx_list[idx2])
            end_time = time()
            runtimes.append(end_time - start_time)
            lchd_by_atom.append(lchd_all)

            chain_chain_mean_locohd_mx[idx1, idx2] = chain_chain_mean_locohd_mx[idx2, idx1] = np.mean(lchd_all)

    lchd_by_atom = np.mean(lchd_by_atom, axis=0)
    print(f"Mean time per run: {np.mean(runtimes):.5f} s")
    print(f"Std of runtimes: {np.std(runtimes):.10f} s")
    print(f"Total runtime: {np.sum(runtimes):.5f} s")

    sorted_dmx_mask = np.argsort(np.mean(chain_chain_mean_locohd_mx, axis=0))
    chain_chain_mean_locohd_mx = chain_chain_mean_locohd_mx[:, sorted_dmx_mask][sorted_dmx_mask, :]

    # Save b-factor labelled structure
    pdb_io = PDBIO()
    prot_structure = PDBParser(QUIET=True).get_structure("", prot_root_path / all_files[0])
    prot_chain = prot_structure[0].child_list[0]

    for lchd_score, (resi_number, atom_name) in zip(lchd_by_atom, accepted_atoms):
        prot_chain[resi_number][atom_name].bfactor = 100 * lchd_score
            
    pdb_io.set_structure(prot_structure)
    pdb_io.save(str(save_dir / f"{save_name}_blabelled.pdb"), select=AtomSelector(accepted_atoms))

    return chain_chain_mean_locohd_mx, lchd_by_atom


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
