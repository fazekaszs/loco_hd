import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from pathlib import Path
from typing import List, Tuple, Dict, Any
from time import time

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import Select

from loco_hd import LoCoHD, WeightFunction, PrimitiveAssigner, PrimitiveAtomTemplate

ATOM_ID = Tuple[int, str]

SAVE_DIR = Path("../../workdir/prot_batch_resuls/h5")
PATHS_AND_NAMES = [
    # ("../../data_sources/pdb_files/h5/dummy", "h5_dummy"),
    ("../../data_sources/pdb_files/h5/277", "h5_277"),
    ("../../data_sources/pdb_files/h5/288", "h5_288"),
    ("../../data_sources/pdb_files/h5/299", "h5_299"),
    ("../../data_sources/pdb_files/h5/310", "h5_310"),
    ("../../data_sources/pdb_files/h5/321", "h5_321"),
    # ("../../data_sources/pdb_files/PED00075e000", "PED00075e000"),
    # ("../../data_sources/pdb_files/PED00072e000", "PED00072e000"),
]
PRIMITIVE_TYPING_SCHEME = "all_atom.config.json"
MM_TO_INCH = 0.0393701


class AtomSelector(Select):

    def __init__(self, accepted_atoms: List[ATOM_ID]):
        super().__init__()
        self.accepted_atoms = accepted_atoms

    def accept_atom(self, atom: Atom):

        resi: Residue = atom.parent
        if (resi.get_id()[1], atom.get_name()) in self.accepted_atoms:
            return True
        return False


def calculate_rmsd(templates1: List[PrimitiveAtomTemplate], templates2: List[PrimitiveAtomTemplate]) -> float:

    # Calculates the optimal 3D alignment using the Kabsch algorithm:
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    # https://towardsdatascience.com/the-definitive-procedure-for-aligning-two-sets-of-3d-points-with-the-kabsch-algorithm-a7ec2126c87e

    coords1 = list(map(lambda x: x.coordinates, templates1))
    coords1 = np.array(coords1)
    coords1 -= np.mean(coords1, axis=0, keepdims=True)

    coords2 = list(map(lambda x: x.coordinates, templates2))
    coords2 = np.array(coords2)
    coords2 -= np.mean(coords2, axis=0, keepdims=True)

    cov_mx = coords1.T @ coords2
    mx_u, mx_s, mx_vt = np.linalg.svd(cov_mx)
    inner_mx = np.eye(3)
    inner_mx[2, 2] = 1 if np.linalg.det(mx_u @ mx_vt) > 0 else -1

    rot_mx = mx_u @ inner_mx @ mx_vt

    new_coords1 = coords1 @ rot_mx

    rmsd = np.sum((new_coords1 - coords2) ** 2, axis=1)
    rmsd = np.sqrt(np.mean(rmsd))

    return rmsd


def plot_result(in_dict: Dict[str, Any]):

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.bottom"] = 0.15

    rmsd_dmx: np.ndarray = in_dict["rmsd_dmx"]
    lchd_dmx: np.ndarray = in_dict["lchd_dmx"]
    lchd_by_atom: np.ndarray = in_dict["lchd_by_atom"]
    dmx_lchd_min: float = in_dict["dmx_lchd_min"]
    dmx_lchd_max: float = in_dict["dmx_lchd_max"]
    save_name: str = in_dict["save_name"]

    fig, ax = plt.subplots(1, 2)

    # Showing the structure - structure distance matrix
    im = ax[0].imshow(lchd_dmx, cmap="coolwarm", vmin=dmx_lchd_min, vmax=dmx_lchd_max)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar_ticks = np.arange(dmx_lchd_min, dmx_lchd_max, 0.025)
    cbar.set_ticks(cbar_ticks, labels=[f"{tick:.1%}" for tick in cbar_ticks])

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("LoCoHD Score of\nthe Structure Pairs")

    # Showing the atomic LoCoHD scores
    ax[1].plot(np.sort(lchd_by_atom), c="black")
    ax[1].set_xlabel("Rank of atom by sorted LoCoHD score")
    ax[1].set_ylabel("LoCoHD score")
    ax[1].set_title("Distribution of\nAtom LoCoHD Scores")

    atom_lchd_mean = np.mean(lchd_by_atom)
    atom_lchd_std = np.std(lchd_by_atom)
    ax[1].plot([0, len(lchd_by_atom)], [atom_lchd_mean, atom_lchd_mean], c="red")
    ax[1].fill_between(
        [0, len(lchd_by_atom)],
        2 * [atom_lchd_mean - atom_lchd_std, ],
        2 * [atom_lchd_mean + atom_lchd_std, ],
        alpha=0.25, color="red", edgecolor=None
    )

    atom_lchd_min = np.min(lchd_by_atom)
    atom_lchd_max = np.max(lchd_by_atom)
    plot_ticks = np.arange(atom_lchd_min, atom_lchd_max, (atom_lchd_max - atom_lchd_min) / 10)
    ax[1].set_yticks(plot_ticks, labels=[f"{tick:.1%}" for tick in plot_ticks])

    # Setting the aspect ratio
    aspect = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
    ax[1].set_aspect(aspect)

    # Adding the statistics as a legend
    atom_lchd_median = np.median(lchd_by_atom)
    legend_labels = list()
    legend_labels.append(f"Min = {atom_lchd_min:.1%}")
    legend_labels.append(f"Max = {atom_lchd_max:.1%}")
    legend_labels.append(f"Mean = {atom_lchd_mean:.1%}")
    legend_labels.append(f"Median = {atom_lchd_median:.1%}")
    legend_labels.append(f"StD = {atom_lchd_std:.1%}")
    legend_handles = Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)
    legend_handles = [legend_handles, ] * len(legend_labels)
    ax[1].legend(
        legend_handles, legend_labels,
        loc="best", fontsize="small", fancybox=True,
        framealpha=0.7, handlelength=0, handletextpad=0
    )

    fig.set_size_inches(180 * MM_TO_INCH, 88 * MM_TO_INCH)
    fig.subplots_adjust(wspace=0.5)
    fig.savefig(SAVE_DIR / f"{save_name}_plot.svg", dpi=300)

    # Plotting the LoCoHD - RMSD relation
    # This goes to a new plot!
    fig, ax = plt.subplots()

    tril_idxs = np.tril_indices(len(rmsd_dmx), k=-1)
    tril_lhcd_dmx = lchd_dmx[tril_idxs]
    tril_rmsd_dmx = rmsd_dmx[tril_idxs]
    ax.scatter(
        tril_lhcd_dmx, tril_rmsd_dmx,
        marker=".", alpha=0.3, color="blue", linewidth=0
    )
    ax.set_xlabel("LoCoHD score")
    ax.set_ylabel("RMSD / $\\AA$")

    spr = spearmanr(tril_rmsd_dmx, tril_lhcd_dmx).correlation
    ax.set_title(f"Comparison of structure-structure\nLoCoHD and RMSD values (SpR: {spr:.5f})")

    tril_lhcd_dmx_min = np.min(tril_lhcd_dmx)
    tril_lhcd_dmx_max = np.max(tril_lhcd_dmx)
    plot_ticks = np.arange(tril_lhcd_dmx_min, tril_lhcd_dmx_max, (tril_lhcd_dmx_max - tril_lhcd_dmx_min) / 6)
    ax.set_xticks(plot_ticks, labels=[f"{tick:.1%}" for tick in plot_ticks])

    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)
    fig.savefig(SAVE_DIR / f"{save_name}_with_rmsd.svg", dpi=300)


def compare_structures(prot_root_path: Path, save_name: str) -> Dict[str, Any]:

    print(f"Starting {save_name}...")

    # Create the PrimitiveAssigner instance
    primitive_assigner = PrimitiveAssigner(Path("../../primitive_typings") / PRIMITIVE_TYPING_SCHEME)

    # Initialize LoCoHD instance
    w_func = WeightFunction("uniform", [3., 10.])
    lchd = LoCoHD(primitive_assigner.all_primitive_types, w_func)

    # Collect all the filenames in the directory
    all_files: List[str] = os.listdir(prot_root_path)
    all_files = list(filter(lambda x: x.endswith(".pdb"), all_files))
    all_files = sorted(all_files)

    print(f"Found {len(all_files)} files!")

    # Read proteins and create the primitive atom template lists
    template_lists: List[List[PrimitiveAtomTemplate]] = list()
    file_name: str
    for file_name in all_files:

        prot_structure = PDBParser(QUIET=True).get_structure("", prot_root_path / file_name)
        primitive_atom_templates = primitive_assigner.assign_primitive_structure(prot_structure)
        template_lists.append(primitive_atom_templates)

    print(f"Proteins read! Primitive types assigned!")

    del file_name, prot_structure, primitive_atom_templates

    # Calculate the RMSD matrix between the structure pairs
    rmsd_dmx = np.zeros((len(template_lists), len(template_lists)))
    for idx1 in range(len(template_lists)):
        for idx2 in range(idx1 + 1, len(template_lists)):
            rmsd_dmx[idx1, idx2] = calculate_rmsd(template_lists[idx1], template_lists[idx2])
            rmsd_dmx[idx2, idx1] = rmsd_dmx[idx1, idx2]

    print(f"RMSD matrix created!")

    # Collect the accepted atoms and the primitive_sequence
    primitive_sequence, accepted_atoms = list(), list()
    for pra_template in template_lists[0]:

        resi_number = pra_template.atom_source.source_residue[3][1]
        atom_name = pra_template.atom_source.source_atom[0]
        accepted_atoms.append((resi_number, atom_name))

        primitive_sequence.append(pra_template.primitive_type)

    del pra_template, resi_number, atom_name

    print(f"Accepted atoms collected!")

    # Collect the contacts that are homo-residue contacts
    homo_contact_idx_pairs: List[Tuple[int, int]] = list()
    for idx1 in range(len(template_lists[0])):
        for idx2 in range(idx1 + 1, len(template_lists[0])):

            resi_id1 = template_lists[0][idx1].atom_source.source_residue
            resi_id2 = template_lists[0][idx2].atom_source.source_residue

            if resi_id1 == resi_id2:
                homo_contact_idx_pairs.append((idx1, idx2))
                homo_contact_idx_pairs.append((idx2, idx1))

    del idx1, idx2, resi_id1, resi_id2

    print(f"Homo-residue contacts ({len(homo_contact_idx_pairs) // 2} in total) collected!")

    # Create the distance matrices
    dmx_list: List[np.ndarray] = list()
    for template_list in template_lists:

        dmx = list()
        for pra_template in template_list:
            dmx.append(pra_template.coordinates)
        dmx = np.array(dmx)
        dmx = dmx[np.newaxis, ...] - dmx[:, np.newaxis, :]
        dmx = np.sqrt(np.sum(dmx ** 2, axis=2))

        # Ban homo-residue contacts
        for idx1, idx2 in homo_contact_idx_pairs:
            dmx[idx1][idx2] = float("inf")

        dmx_list.append(dmx)

    del idx1, idx2, template_list, pra_template, dmx

    print(f"Distance matrices created! Starting LoCoHD calculations...")

    # Calculate lchd mean and std values
    n_of_structures = len(dmx_list)
    n_of_comparisons = n_of_structures * (n_of_structures - 1) / 2
    lchd_dmx = np.zeros((n_of_structures, n_of_structures))
    runtimes = list()
    lchd_by_atom = list()
    for idx1 in range(n_of_structures):
        for idx2 in range(idx1 + 1, n_of_structures):

            start_time = time()

            # The magic happens here
            lchd_all = lchd.from_dmxs(
                primitive_sequence,
                primitive_sequence,
                dmx_list[idx1],
                dmx_list[idx2]
            )

            end_time = time()
            runtimes.append(end_time - start_time)

            lchd_by_atom.append(lchd_all)
            lchd_dmx[idx1, idx2] = lchd_dmx[idx2, idx1] = np.mean(lchd_all)

            print(f"\rCompletion: {len(runtimes) / n_of_comparisons:.1%}", end="")

    print()
    lchd_by_atom = np.mean(lchd_by_atom, axis=0)

    # Printing time statistics
    print(f"Mean time per run: {np.mean(runtimes):.5f} s")
    print(f"Std of runtimes: {np.std(runtimes):.10f} s")
    print(f"Total runtime: {np.sum(runtimes):.5f} s")

    # Sort the rows and columns of the locohd and rmsd distance matrices
    #  based on the mean row lchd values.
    sorted_dmx_mask = np.argsort(np.mean(lchd_dmx, axis=0))
    lchd_dmx = lchd_dmx[:, sorted_dmx_mask][sorted_dmx_mask, :]
    rmsd_dmx = rmsd_dmx[:, sorted_dmx_mask][sorted_dmx_mask, :]

    # Save raw dmx data
    raw_save_name = save_name + "_rawDmxData.pickle"
    with open(SAVE_DIR / raw_save_name, "wb") as f:
        pickle.dump({
            "lchd_dmx": lchd_dmx,
            "rmsd_dmx": rmsd_dmx,
            "filenames": np.array(all_files)[sorted_dmx_mask]
        }, f)
    print("Raw distance matrix data succesfully saved!")

    # Save b-factor labelled structure
    b_labelled_pdb = primitive_assigner.generate_primitive_pdb(template_lists[0], b_labels=lchd_by_atom)

    pdb_save_name = save_name + "_blabelled.pdb"
    with open(SAVE_DIR / pdb_save_name, "w") as f:
        f.write(b_labelled_pdb)

    print(f"B-labelled primitive structure saved as {pdb_save_name} based on the template {all_files[0]}!")

    return {"rmsd_dmx": rmsd_dmx, "lchd_dmx": lchd_dmx, "lchd_by_atom": lchd_by_atom}


def main():

    rmsd_dmxs, lchd_dmxs, lchd_by_atom = list(), list(), list()

    for prot_root_path, prot_name in PATHS_AND_NAMES:

        out_dict = compare_structures(Path(prot_root_path), prot_name)
        rmsd_dmxs.append(out_dict["rmsd_dmx"])
        lchd_dmxs.append(out_dict["lchd_dmx"])
        lchd_by_atom.append(out_dict["lchd_by_atom"])

    print("Starting to plot!")
    dmx_lchd_min = np.min(lchd_dmxs)
    dmx_lchd_max = np.max(lchd_dmxs)

    for idx in range(len(PATHS_AND_NAMES)):

        plot_result_input = {
            "rmsd_dmx": rmsd_dmxs[idx],
            "lchd_dmx": lchd_dmxs[idx],
            "lchd_by_atom": lchd_by_atom[idx],
            "dmx_lchd_min": dmx_lchd_min,
            "dmx_lchd_max": dmx_lchd_max,
            "save_name": PATHS_AND_NAMES[idx][1]
        }

        plot_result(plot_result_input)


if __name__ == "__main__":
    main()
