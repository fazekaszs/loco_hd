import MDAnalysis as mda
from MDAnalysis.coordinates.base import Timestep
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from time import time
from pathlib import Path

import numpy as np
from loco_hd import LoCoHD, PrimitiveAtom

from sklearn.decomposition import PCA

TERMINAL_O = ["OT1", "OT2", "OC1", "OC2", "OXT"]
PRIMITIVE_TYPES = ["O_neg", "O_neu", "N_pos", "N_neu", "C_ali", "C_aro", "S"]
SIDECHAIN_ATOMS = [
        ["GLU:OE1", "GLU:OE2", "ASP:OD1", "ASP:OD2"],
        ["GLN:OE1", "ASN:OD1", "SER:OG", "THR:OG1", "TYR:OH"],
        ["ARG:NE", "ARG:NH1", "ARG:NH2", "LYS:NZ"],  # old arginine primitives
        # ["ARG:NE", "LYS:NZ"],  # new arginine primitives
        ["GLN:NE2", "ASN:ND2", "HIS:ND1", "HIS:NE2", "TRP:NE1"],  # old arginine primitives
        # ["GLN:NE2", "ASN:ND2", "HIS:ND1", "HIS:NE2", "TRP:NE1", "ARG:NH1", "ARG:NH2"],  # new arginine primitives
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


def universe_to_primitives(frame: Timestep, universe: mda.Universe, atom_mask: np.ndarray):

    resi_posi = [np.mean(frame.positions[resi_atom_idxs], axis=0) for resi_atom_idxs in universe.residues.indices]
    resi_sources = [f"{resi.segindex}/{resi.resindex}-{resi.resname}"
                    for idx, resi in enumerate(universe.residues)]
    prim_atoms = [PrimitiveAtom("Cent", source, posi) for source, posi in zip(resi_sources, resi_posi)]
    anchors = list(range(len(prim_atoms)))

    atom_posi = frame.positions[atom_mask]
    atom_type = [assign_primitive_type(atom.residue.resname, atom.name)
                 for idx, atom in enumerate(universe.atoms)
                 if atom_mask[idx]]
    atom_sources = [f"{atom.segindex}/{atom.residue.resindex}-{atom.residue.resname}"
                    for idx, atom in enumerate(universe.atoms)
                    if atom_mask[idx]]

    prim_atoms += [PrimitiveAtom(pr_type, source, posi)
                   for pr_type, source, posi in zip(atom_type, atom_sources, atom_posi)]

    return prim_atoms, anchors


def arg_median(data: np.ndarray):
    return np.argsort(data)[len(data) // 2]


def main():

    trajectory_path = "/home/fazekaszs/gmx_tmp/21F-NH2_opcMDnjmc_lowsample.xtc"
    structure_path = "/home/fazekaszs/gmx_tmp/21F-NH2_opcMD_onlyProt.tpr"
    workdir = Path("./workdir/trajectory_analysis")

    universe = mda.Universe(structure_path, trajectory_path)
    heavy_atom_mask = universe.atoms.elements != "H"
    lchd = LoCoHD(PRIMITIVE_TYPES + ["Cent", ], ("uniform", [3, 10]))
    delta_frame = 50

    prim_atoms1, anchors = universe_to_primitives(universe.trajectory[0], universe, heavy_atom_mask)
    anchors = [(x, x) for x in anchors]

    full_trajectory_length = len(universe.trajectory[::delta_frame])
    all_points = list()
    start_real_time = time()

    frame: Timestep
    for frame_idx, frame in enumerate(universe.trajectory[::delta_frame]):

        prim_atoms2, _ = universe_to_primitives(frame, universe, heavy_atom_mask)
        lchd_scores = lchd.from_primitives(prim_atoms1, prim_atoms2, anchors, True, 10)
        all_points.append(lchd_scores)

        if frame_idx % 100 == 0:
            current_real_time = time()
            eta = full_trajectory_length - frame_idx  # number of remaining frames
            eta *= (current_real_time - start_real_time) / (frame_idx + 1)  # current time / frame rate
            print(f"{frame_idx / full_trajectory_length:.1%} at time {frame.time:.0f} ps. ETA: {eta:.1f} s.")

    print("Calculations done! Starting to plot...")
    all_points = np.array(all_points)
    np.save(str(workdir / "LoCoHD_scores.npy"), all_points)

    # Plotting of individual LoCoHD time dependency
    fig, ax = plt.subplots()
    plt.tight_layout()
    x_values = np.arange(len(all_points)) * delta_frame * universe.trajectory.dt / 1000
    for plot_idx in range(all_points.shape[1]):

        ax.cla()
        resi = universe.residues[plot_idx]
        plot_title = f"{resi.segindex}.{resi.resindex}-{resi.resname}"
        y_values = all_points[:, plot_idx]

        max_lchd_idx = np.argmax(all_points[:, plot_idx])
        max_lchd_time = max_lchd_idx * delta_frame * universe.trajectory.dt
        max_lchd_score = all_points[max_lchd_idx, plot_idx]

        median_lchd_idx = arg_median(all_points[:, plot_idx])
        median_lchd_time = median_lchd_idx * delta_frame * universe.trajectory.dt
        median_lchd_score = all_points[median_lchd_idx, plot_idx]

        legend_labels = list()
        legend_labels.append(f"Max score: {max_lchd_score:.1%} at time: {max_lchd_time:.0f} ps")
        legend_labels.append(f"Median score: {median_lchd_score:.1%} at time: {median_lchd_time:.0f} ps")

        legend_handles = Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)
        legend_handles = [legend_handles, ] * len(legend_labels)

        ax.legend(legend_handles, legend_labels,
                  loc="lower right", fontsize="small", fancybox=True,
                  framealpha=0.7, handlelength=0, handletextpad=0)
        ax.plot(x_values, y_values, c="black")
        ax.set_xlabel("$t$ / ns")
        ax.set_ylabel("LoCoHD score")
        ax.set_title(plot_title)
        fig.savefig(str(workdir / f"{plot_title}.png"), dpi=200, bbox_inches="tight")

    # Plotting the first principal component time dependency
    principal_comp = PCA(n_components=1).fit_transform(all_points)
    ax.cla()
    ax.plot(x_values, principal_comp[:, 0], c="black")
    ax.set_xlabel("$t$ / ns")
    ax.set_ylabel("PCA1")
    ax.set_title("Time Evolution of the First Principal Component")
    fig.savefig(workdir / "pca.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
