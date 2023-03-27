import warnings
from time import time
from typing import List
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA

import MDAnalysis as mda
from MDAnalysis import Universe
from MDAnalysis.coordinates.base import Timestep

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from loco_hd import LoCoHD, PrimitiveAtom
from atom_converter_utils import PrimitiveAssigner, PrimitiveAtomTemplate, PrimitiveAtomSource

warnings.filterwarnings("ignore")


class MDPrimitiveAssigner(PrimitiveAssigner):
    def assign_from_universe(self, frame: Timestep, universe: Universe) -> List[PrimitiveAtomTemplate]:

        out = list()

        resi: mda.core.groups.Residue
        for resi in universe.residues:

            resi_name = resi.resname
            resi_id = ("prot", 0, "A", (" ", resi.ix, " "))

            for tse in self.scheme:

                if not tse.match_resi(resi_name):
                    continue

                atom_names = list()
                atom_coords = list()

                atom: mda.core.groups.Atom
                for atom in resi.atoms:

                    if not tse.match_atom(atom.name):
                        continue

                    atom_names.append(atom.name)
                    atom_coords.append(frame.positions[atom.ix])

                if len(atom_coords) == 0:
                    continue

                centroid = np.mean(atom_coords, axis=0)
                pras = PrimitiveAtomSource(resi_id, resi_name, atom_names)
                prat = PrimitiveAtomTemplate(tse.primitive_type, centroid, pras)
                out.append(prat)

        return out


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def arg_median(data: np.ndarray):
    return np.argsort(data)[len(data) // 2]


def main():

    # Establish working directory and data sources
    workdir = Path("./workdir/trajectory_analysis/ttr_forLoCoHD")
    trajectory_path = workdir / "TTR_WT_opcMD_mcc_dt100.xtc"
    structure_path = workdir / "TTR_WT_opcMD_grompp_onlyProt.tpr"
    primitive_typing_scheme_path = Path("./primitive_typings/coarse_grained_with_centroid.config.json")

    # Read the structure file and the trajectory
    universe = mda.Universe(str(structure_path), str(trajectory_path))

    # Define the primitive typing scheme
    primitive_assigner = MDPrimitiveAssigner(primitive_typing_scheme_path)

    # Define the LoCoHD instance
    lchd = LoCoHD(primitive_assigner.all_primitive_types, ("uniform", [3, 10]))

    # Get the primitive atoms for the first frame and also define the anchor atoms
    pra_templates_start = primitive_assigner.assign_from_universe(universe.trajectory[0], universe)
    prim_atoms_start = list(map(prat_to_pra, pra_templates_start))
    anchors = [(idx, idx) for idx, prat in enumerate(pra_templates_start) if prat.primitive_type == "Cent"]

    # Main loop
    delta_frame = 25
    full_trajectory_length = len(universe.trajectory[::delta_frame])
    all_points = list()
    start_real_time = time()

    frame: Timestep
    for frame_idx, frame in enumerate(universe.trajectory[::delta_frame]):

        # Get the primitive atoms for the current frame
        pra_templates = primitive_assigner.assign_from_universe(frame, universe)
        prim_atoms = list(map(prat_to_pra, pra_templates))

        # Calculate the LoCoHD scores
        lchd_scores = lchd.from_primitives(prim_atoms_start, prim_atoms, anchors, True, 10)
        all_points.append(lchd_scores)

        # Print out time statistics
        current_real_time = time()
        eta = full_trajectory_length - frame_idx  # number of remaining frames
        eta *= (current_real_time - start_real_time) / (frame_idx + 1)  # current time / frame rate
        print(f"\r{frame_idx / full_trajectory_length:.1%} at time {frame.time:.0f} ps. ETA: {eta:.1f} s.", end="")

    print("\nCalculations done! Starting to plot...")
    all_points = np.array(all_points)
    np.save(str(workdir / "LoCoHD_scores.npy"), all_points)

    # Plotting of individual LoCoHD time dependencies
    fig, ax = plt.subplots()
    plt.tight_layout()
    x_values = np.arange(len(all_points)) * delta_frame * universe.trajectory.dt / 1000
    max_lchd = np.max(all_points)
    y_axis_ticks = np.arange(0, max_lchd, max_lchd / 10)

    for plot_idx in range(all_points.shape[1]):

        ax.cla()
        resi = universe.residues[plot_idx]
        chain_id = "ABCDEFGHIJKLMNOPQRSTUOVWXYZ"[resi.segindex]
        plot_title = f"{chain_id}.{resi.resindex + 1}-{resi.resname}"
        y_values = all_points[:, plot_idx]

        print(f"\rPlotting \"{plot_title}\"", end="")

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
        ax.set_yticks(y_axis_ticks, labels=[f"{tick:.1%}" for tick in y_axis_ticks])
        ax.set_ylim(0, max_lchd)
        fig.savefig(str(workdir / f"{plot_title}.png"), dpi=200, bbox_inches="tight")

    print("\nPlotting for residues done! Plotting principal components...")

    # Plotting the first principal component's time dependency
    # (Skip the first frame, since it is an outlier)
    principal_comp = PCA(n_components=2).fit_transform(all_points[1:])

    ax.cla()
    ax.plot(x_values[1:], principal_comp[:, 0], c="black")
    ax.set_xlabel("$t$ / ns")
    ax.set_ylabel("PCA1")
    ax.set_title("Time Evolution of the First Principal Component")
    fig.savefig(workdir / "pca.png", dpi=200, bbox_inches="tight")

    # Plotting the first two principal component's time dependency
    ax.cla()
    ax.scatter(principal_comp[:, 0], principal_comp[:, 1], c=x_values[1:])
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("Time Evolution of the\nFirst Two Principal Components")
    fig.savefig(workdir / "pca_2.png", dpi=200, bbox_inches="tight")

    print("Principal components plotted! Saving b-factor labelled pdb file...")

    # Saving a b-factor labelled structure
    universe.add_TopologyAttr("tempfactors")
    tempfactors = np.std(all_points[1:], axis=0)

    print(f"Created {len(tempfactors)} number of temperature factors...")

    resi: mda.core.groups.Residue
    for resi, t_value in zip(universe.residues, tempfactors):
        resi.atoms.tempfactors += t_value

    universe.select_atoms("protein").write(str(workdir / "b_labelled.pdb"))


if __name__ == "__main__":
    main()
