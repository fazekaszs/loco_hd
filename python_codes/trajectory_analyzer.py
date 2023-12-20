import os
import warnings
from time import time
from typing import List
from pathlib import Path

import numpy as np

from scipy.stats import skew, kurtosis

from sklearn.decomposition import PCA

import MDAnalysis as mda
from MDAnalysis import Universe
from MDAnalysis.coordinates.base import Timestep

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from loco_hd import LoCoHD, PrimitiveAtom, WeightFunction, PrimitiveAssigner, PrimitiveAtomTemplate, \
    PrimitiveAtomSource, TagPairingRule

warnings.filterwarnings("ignore")

# Establish working directory, data sources
SOURCE_DIR = Path("../data_sources/trajectory_analysis/podocin_dimer")
TARGET_DIR = Path("../workdir/trajectory_analysis/podocin_dimer")
TRAJECTORY_PATH = SOURCE_DIR / "podocin_wt_dimer_mcc_dt100.xtc"
STRUCTURE_PATH = SOURCE_DIR / "podocin_wt_dimer_onlyProt.tpr"
PRIMITIVE_TYPING_SCHEME_PATH = Path("../primitive_typings/coarse_grained_with_centroid.config.json")
MM_TO_INCH = 0.0393701


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

                if tse.atom_counter == "any":
                    pass
                elif tse.atom_counter == len(atom_coords):
                    pass
                else:
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


def calculate_lchd_scores(universe: Universe, delta_frame: int) -> np.ndarray:

    print("Calculating the time-dependency of residue LoCoHD scores!")

    # Define the primitive typing scheme
    primitive_assigner = MDPrimitiveAssigner(PRIMITIVE_TYPING_SCHEME_PATH)

    # Define the LoCoHD instance
    w_func = WeightFunction("uniform", [3, 10])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    lchd = LoCoHD(primitive_assigner.all_primitive_types, w_func, tag_pairing_rule)

    # Get the primitive atoms for the first frame and also define the anchor atoms
    pra_templates_start = primitive_assigner.assign_from_universe(universe.trajectory[0], universe)
    prim_atoms_start = list(map(prat_to_pra, pra_templates_start))
    anchors = [(idx, idx) for idx, prat in enumerate(pra_templates_start) if prat.primitive_type == "Cent"]

    # Main loop
    all_points = list()
    full_trajectory_length = len(universe.trajectory[::delta_frame])
    start_real_time = time()

    frame: Timestep
    for frame_idx, frame in enumerate(universe.trajectory[::delta_frame]):

        # Get the primitive atoms for the current frame
        pra_templates = primitive_assigner.assign_from_universe(frame, universe)
        prim_atoms = list(map(prat_to_pra, pra_templates))

        # Calculate the LoCoHD scores
        lchd_scores = lchd.from_primitives(prim_atoms_start, prim_atoms, anchors, 10)
        all_points.append(lchd_scores)

        # Print out time statistics
        current_real_time = time()
        eta = full_trajectory_length - frame_idx  # number of remaining frames
        eta *= (current_real_time - start_real_time) / (frame_idx + 1)  # current time / frame rate
        print(f"\r{frame_idx / full_trajectory_length:.1%} at time {frame.time:.0f} ps. ETA: {eta:.1f} s.", end="")

    print("\nCalculations done!")
    return np.array(all_points)


def calculate_bimodality_coeff(universe: Universe, all_points: np.ndarray):

    print("Calculating bimodalities!")

    lchd_skewnesses = skew(all_points, axis=0)
    lchd_kurtoses = kurtosis(all_points, axis=0, fisher=False)

    sarle_bimodality_coeff = (lchd_skewnesses ** 2 + 1) / lchd_kurtoses
    bimodality_order = np.argsort(sarle_bimodality_coeff)[::-1]
    residues = [f"{resi.segindex}.{resi.ix + 1}-{resi.resname}" for resi in universe.residues]
    residues = np.array(residues)[bimodality_order]
    sarle_bimodality_coeff = sarle_bimodality_coeff[bimodality_order]

    out = ""
    for resname, coeff in zip(residues, sarle_bimodality_coeff):
        out += f"{resname}\t{coeff}\n"
    out = out[:-1]

    with open(TARGET_DIR / "resi_bimodalities.txt", "w") as f:
        f.write(out)


def plot_time_dependencies(
        universe: Universe,
        x_values: np.ndarray,
        all_points: np.ndarray,
        delta_frame: int
):

    print("Plotting LoCoHD score time dependencies!")

    fig, ax = plt.subplots()
    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)

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
        max_lchd_time += universe.trajectory[0].time
        max_lchd_score = all_points[max_lchd_idx, plot_idx]

        median_lchd_idx = arg_median(all_points[:, plot_idx])
        median_lchd_time = median_lchd_idx * delta_frame * universe.trajectory.dt
        median_lchd_time += universe.trajectory[0].time
        median_lchd_score = all_points[median_lchd_idx, plot_idx]

        mean_lchd_score = np.mean(all_points[:, plot_idx])
        std_lchd_score = np.std(all_points[:, plot_idx])

        legend_labels = list()
        legend_labels.append(f"Mean score: {mean_lchd_score:.1%}")
        legend_labels.append(f"StDev of the score: {std_lchd_score:.1%}")
        legend_labels.append(f"Median score: {median_lchd_score:.1%} at time: {median_lchd_time:.0f} ps")
        legend_labels.append(f"Max score: {max_lchd_score:.1%} at time: {max_lchd_time:.0f} ps")

        legend_handles = Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)
        legend_handles = [legend_handles, ] * len(legend_labels)

        ax.legend(legend_handles, legend_labels,
                  loc="upper left", fontsize="small", fancybox=True,
                  framealpha=0.7, handlelength=0, handletextpad=0)
        ax.plot(x_values, y_values, c="black")
        ax.set_xlabel("$t$ / ns")
        ax.set_ylabel("LoCoHD score")
        ax.set_title(plot_title)
        ax.set_yticks(y_axis_ticks, labels=[f"{tick:.1%}" for tick in y_axis_ticks])
        ax.set_ylim(0, max_lchd)
        fig.savefig(str(TARGET_DIR / f"{plot_title}.svg"), dpi=300)

    # Plotting of the mean (along residues) LoCoHD value's time dependency
    mean_lchd_t = np.mean(all_points, axis=1)
    ax.cla()

    max_lchd_idx = np.argmax(mean_lchd_t)
    max_lchd_time = max_lchd_idx * delta_frame * universe.trajectory.dt
    max_lchd_time += universe.trajectory[0].time
    max_lchd_score = mean_lchd_t[max_lchd_idx]

    median_lchd_idx = arg_median(mean_lchd_t)
    median_lchd_time = median_lchd_idx * delta_frame * universe.trajectory.dt
    median_lchd_time += universe.trajectory[0].time
    median_lchd_score = mean_lchd_t[median_lchd_idx]

    mean_lchd_score = np.mean(mean_lchd_t)
    std_lchd_score = np.std(mean_lchd_t)

    legend_labels = list()
    legend_labels.append(f"Mean score: {mean_lchd_score:.1%}")
    legend_labels.append(f"StDev of the score: {std_lchd_score:.1%}")
    legend_labels.append(f"Median score: {median_lchd_score:.1%} at time: {median_lchd_time:.0f} ps")
    legend_labels.append(f"Max score: {max_lchd_score:.1%} at time: {max_lchd_time:.0f} ps")

    legend_handles = Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)
    legend_handles = [legend_handles, ] * len(legend_labels)

    ax.legend(legend_handles, legend_labels,
              loc="upper left", fontsize="small", fancybox=True,
              framealpha=0.7, handlelength=0, handletextpad=0)
    ax.plot(x_values, mean_lchd_t, c="black")
    ax.set_xlabel("$t$ / ns")
    ax.set_ylabel("LoCoHD score")
    ax.set_title("Mean LoCoHD values")
    ax.set_yticks(y_axis_ticks, labels=[f"{tick:.1%}" for tick in y_axis_ticks])
    ax.set_ylim(0, max_lchd)
    fig.savefig(str(TARGET_DIR / f"mean_locohd.svg"), dpi=300)

    print("\nPlotting for residues done!")


def plot_pca(x_values: np.ndarray, all_points: np.ndarray):

    print("Calculating and plotting LoCoHD principal components...")

    fig, ax = plt.subplots()
    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)

    # Skip the first frame, since it is an outlier (always 0)
    pca = PCA()
    principal_comp = pca.fit_transform(all_points[1:])

    # Plotting the first principal component's time dependency
    ax.cla()
    ax.plot(x_values[1:], principal_comp[:, 0], c="black")
    ax.set_xlabel("$t$ / ns")
    ax.set_ylabel("PCA1")
    ax.set_title("Time Evolution of the First Principal Component")
    fig.savefig(TARGET_DIR / "pca.svg", dpi=300)

    # Plotting the first two principal component's time dependency
    ax.cla()
    ax.scatter(
        principal_comp[:, 0], principal_comp[:, 1],
        c=x_values[1:], cmap="autumn",
        marker=".", edgecolors="black"
    )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("Time Evolution of the\nFirst Two Principal Components")
    fig.savefig(TARGET_DIR / "pca_2.svg", dpi=300)

    # Plotting the explained variance ratio of each component
    ax.cla()
    ax.set_xlabel("Rank of component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Cumulative Explained Variance of\neach Principal Component")

    y_axis_ticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(y_axis_ticks, labels=[f"{tick:.1%}" for tick in y_axis_ticks])
    cumulative_expl_var = np.cumsum(pca.explained_variance_ratio_)
    cumulative_expl_var = np.append([0, ], cumulative_expl_var)
    ax.plot(np.arange(len(cumulative_expl_var)), cumulative_expl_var, c="black")
    fig.savefig(TARGET_DIR / "pca_explained_variance.svg", dpi=300)

    # Plotting the covariance matrix of the residues
    ax.cla()
    ax.imshow(pca.get_covariance())
    ax.set_title("Covariance Matrix between the\nLoCoHD Scores of Residues")
    fig.savefig(TARGET_DIR / "pca_cov.svg", dpi=300)

    print("Principal components plotted!")


def save_blabelled_pdb(universe: Universe, all_points: np.ndarray):

    print("Creating and Saving b-factor labelled pdb file")

    universe.add_TopologyAttr("tempfactors")
    tempfactors = np.std(all_points[1:], axis=0) * 100

    print(f"Created {len(tempfactors)} number of temperature factors...")

    resi: mda.core.groups.Residue
    for resi, t_value in zip(universe.residues, tempfactors):
        resi.atoms.tempfactors += t_value

    universe.select_atoms("protein").write(str(TARGET_DIR / "b_labelled.pdb"))


def main():

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.bottom"] = 0.15

    # Read the structure file and the trajectory
    universe = mda.Universe(str(STRUCTURE_PATH), str(TRAJECTORY_PATH))

    # Load in or calculate the LoCoHD data
    delta_frame = 25
    scores_path = TARGET_DIR / "LoCoHD_scores.npy"
    if os.path.exists(scores_path):
        print("Already calculated scores found in the working directory! I will use them for the analysis!")
        all_points = np.load(str(scores_path))
    else:
        all_points = calculate_lchd_scores(universe, delta_frame)
        np.save(str(scores_path), all_points)

    # Set the time points at which we made the analysis
    x_values = np.arange(len(all_points)) * delta_frame * universe.trajectory.dt / 1000
    x_values += universe.trajectory[0].time / 1000

    # Calculate the bimodality coefficient of the LoCoHD time dependence for every residue
    calculate_bimodality_coeff(universe, all_points)

    # Plotting the first principal component's time dependency
    plot_pca(x_values, all_points)

    # Saving a b-factor labelled structure
    save_blabelled_pdb(universe, all_points)

    # Plotting of individual LoCoHD time dependencies
    plot_time_dependencies(universe, x_values, all_points, delta_frame)


if __name__ == "__main__":
    main()
