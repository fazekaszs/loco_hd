import os
import sys
import subprocess as subp
import pickle

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

RING_FILE_PATH = Path("../ring-3.0.0/ring/bin/ring")
PISCES_DIR_PATH = Path("../../../../PycharmProjects/databases/pisces_220222")
OUTPUT_PATH = Path("../../workdir/pisces/ring_result")
PISCES_LOCOHD_FILE_PATH = Path("../../workdir/pisces/run_2023-02-08-12-50-23/locohd_data.pisces")

INTERACTIONS = [
    "HBOND", "VDW", "SSBOND", "IONIC", "PIPISTACK", "PICATION"
]
RESI_TLCS = [
    "GLY", "ALA", "VAL", "ILE", "LEU", "PHE",
    "SER", "THR", "TYR", "ASP", "GLU", "ASN",
    "GLN", "TRP", "HIS", "MET", "PRO", "CYS",
    "ARG", "LYS"
]
MM_TO_INCH = 0.0393701


class EnvironmentPairList:

    def __init__(self, ring_data: Dict[str, np.ndarray], locohd_data: List[Tuple[str, str, float]]):

        self.training_data = None  # cached training data

        self.side1_ids = list()
        self.side2_ids = list()
        self.side1_interactions = list()
        self.side2_interactions = list()
        self.locohd_values = list()

        for side1_id, side2_id, locohd_score in locohd_data:

            side1_interaction = ring_data.get(side1_id, np.zeros(len(INTERACTIONS)))
            side2_interaction = ring_data.get(side2_id, np.zeros(len(INTERACTIONS)))

            self.side1_ids.append(side1_id)
            self.side2_ids.append(side2_id)
            self.side1_interactions.append(side1_interaction)
            self.side2_interactions.append(side2_interaction)
            self.locohd_values.append(locohd_score)

    def __len__(self):
        return len(self.locohd_values)

    def get_training_data(self, forced=False):

        if self.training_data is not None and not forced:
            return self.training_data  # if cached data is available, return it

        t_in1, t_in2 = list(), list()
        for idx in range(len(self)):

            side1_tlc_onehot = tlc_to_one_hot(self.side1_ids[idx][-3:])
            side2_tlc_onehot = tlc_to_one_hot(self.side2_ids[idx][-3:])

            # feature vec = residue one-hot + contacts
            side1_feature_vec = np.concatenate([side1_tlc_onehot, self.side1_interactions[idx]])
            side2_feature_vec = np.concatenate([side2_tlc_onehot, self.side2_interactions[idx]])

            t_in1.append(side1_feature_vec)
            t_in2.append(side2_feature_vec)

        # two input tensors, one output tensor
        return np.array(t_in1), np.array(t_in2), np.array(self.locohd_values)


def run_ring() -> None:
    """
    Runs RING on every pdb file inside a (PISCES) dictionary.
    """

    # Collecting input pdb file names
    pdb_file_names = os.listdir(PISCES_DIR_PATH)
    pdb_file_names = list(filter(lambda x: x.endswith(".pdb"), pdb_file_names))

    # Creating RING input file
    ring_input_file_name = "ring_pdb_list.txt"
    ring_input_str = "\n".join([
        str((PISCES_DIR_PATH / pdb_file_name).resolve())
        for pdb_file_name in pdb_file_names
    ])

    with open(OUTPUT_PATH / ring_input_file_name, "w") as f:
        f.write(ring_input_str)

    # Running RING analysis with the input file
    to_subp_run = [
        str(RING_FILE_PATH),
        "-I", str(OUTPUT_PATH / ring_input_file_name),
        "--out_dir", str(OUTPUT_PATH)
    ]

    exit_code = subp.Popen(to_subp_run).wait()

    if exit_code != 0:
        sys.exit(f"Nonzero exitcode encountered in RING: {exit_code}!")

    # Cleanup
    ring_node_files = os.listdir(OUTPUT_PATH)
    ring_node_files = list(filter(lambda x: x.endswith("_ringNodes"), ring_node_files))
    for file_name in ring_node_files:
        os.system(f"rm {str(OUTPUT_PATH.resolve() / file_name)}")


def process_ring_result() -> Dict[str, np.ndarray]:
    """
    Read the ringEdges files and collect the data from them. The output is a dictionary with residue
    specifying keys (pdb_id/chain_id/resi_id-resi_type) and interaction count vectors.

    :return: The residue contact count dictionary.
    """

    ring_edge_file_names = os.listdir(OUTPUT_PATH)
    ring_edge_file_names = list(filter(lambda x: x.endswith(".pdb_ringEdges"), ring_edge_file_names))

    out = dict()
    for file_name in ring_edge_file_names:

        pdb_code = file_name.split(".")[0]

        with open(OUTPUT_PATH / file_name, "r") as f:
            data = f.read()

        data = data.split("\n")[1:]
        data = list(filter(lambda x: len(x) != 0, data))

        for line in data:

            line = line.split("\t")

            side1 = line[0].split(":")
            side1 = f"{pdb_code}/{side1[0]}/{side1[1]}-{side1[3]}"

            side2 = line[2].split(":")
            side2 = f"{pdb_code}/{side2[0]}/{side2[1]}-{side2[3]}"

            interaction = line[1].split(":")[0]

            counts1 = out.get(side1, np.zeros(len(INTERACTIONS)))
            counts1[INTERACTIONS.index(interaction)] += 1
            out[side1] = counts1

            counts2 = out.get(side2, np.zeros(len(INTERACTIONS)))
            counts2[INTERACTIONS.index(interaction)] += 1
            out[side2] = counts2

        print("\rNumber of residues:", len(out), end="")

    # Cleanup
    ring_edge_files = os.listdir(OUTPUT_PATH)
    ring_edge_files = list(filter(lambda x: x.endswith("_ringEdges"), ring_edge_files))
    for file_name in ring_edge_files:
        os.system(f"rm {str(OUTPUT_PATH.resolve() / file_name)}")

    return out


def get_ring_data() -> Dict[str, np.ndarray]:
    """
    Calls the run_ring and process_ring_result functions if necessary, or loads the corresponding, already
    saved ring datafile. Returns a dictionary with residue ids as keys and interaction count vectors as
    values. For the key formats see the process_ring_result function documentation.

    :return: The residue interaction count dictionary.
    """

    ring_out_filename = "collected.pickle"

    if os.path.exists(OUTPUT_PATH / ring_out_filename):

        print(f"{str(OUTPUT_PATH / ring_out_filename)} already exists! Using this file...")
        with open(OUTPUT_PATH / ring_out_filename, "rb") as f:
            ring_data: Dict[str, np.ndarray] = pickle.load(f)

    else:

        run_ring()
        ring_data = process_ring_result()
        with open(OUTPUT_PATH / ring_out_filename, "wb") as f:
            pickle.dump(ring_data, f)

    return ring_data


def tlc_to_one_hot(tlc: str):
    """
    Converts the three-letter code of an amino acid to a one-hot encoding.

    :param tlc: The three-letter code of the amino acid.
    :return: The one-hot encoded vector.
    """

    out = np.zeros(len(RESI_TLCS))
    out[RESI_TLCS.index(tlc)] = 1

    return out


def merge_datasets(ring_data: Dict[str, np.ndarray]) -> EnvironmentPairList:
    """
    Merge the LoCoHD dataset (resi1, resi2, lchd score) and the RING dataset (resi keys, contact count vectors).
    Creates the full dataset for the training of a neural network.

    :param ring_data:
    :return:
    """

    with open(PISCES_LOCOHD_FILE_PATH, "rb") as f:
        locohd_data: List[Tuple[str, str, float]] = pickle.load(f)

    merged_data = EnvironmentPairList(ring_data, locohd_data)

    return merged_data


def create_ffnn():
    """
    Creates a siamese feedforward neural network model. It should have two inputs and must be symmetric
    to input swapping, since it will try to mimic a metric.

    :return: The neural network model.
    """

    # Define layers

    l_in1 = tf.keras.layers.Input(shape=(26, ))
    l_in2 = tf.keras.layers.Input(shape=(26, ))

    l_siamese1 = tf.keras.layers.Dense(256, activation="relu")
    l_siamese2 = tf.keras.layers.Dense(128, activation="relu")

    l_combiner = tf.keras.layers.Lambda(
        lambda x: (x[0] - x[1]) ** 2
    )

    l_ff = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
        tf.keras.layers.Reshape(tuple())
    ])

    # Define tensor paths

    t_vec1 = l_siamese1(l_in1)
    t_vec1 = l_siamese2(t_vec1)

    t_vec2 = l_siamese1(l_in2)
    t_vec2 = l_siamese2(t_vec2)

    t_combined = l_combiner([t_vec1, t_vec2])
    t_combined = l_ff(t_combined)

    # Define and compile model

    ffnn = tf.keras.Model(inputs=(l_in1, l_in2), outputs=t_combined)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    bce = tf.keras.losses.BinaryCrossentropy(name="bce")
    mse = tf.keras.losses.MeanSquaredError(name="mse")
    mae = tf.keras.losses.MeanAbsoluteError(name="mae")
    ffnn.compile(optimizer=optimizer, loss=bce, metrics=[mse, mae])

    return ffnn


def test_interaction_dependence(feature_vecs: np.ndarray, ffnn: tf.keras.Model):

    out_matrix = list()

    # loop over all residue types
    for idx, resname in enumerate(RESI_TLCS):

        resi_mask = feature_vecs[:, idx] == 1
        interaction_mask = np.max(feature_vecs[resi_mask, -len(INTERACTIONS):], axis=0) != 0

        t_in1 = np.zeros(len(RESI_TLCS) + len(INTERACTIONS), dtype=float)
        t_in1[idx] = 1.  # a residue without interactions
        t_in1 = t_in1[np.newaxis, ...]  # add batch dim

        t_in2 = list()
        # loop over all interactions
        for interaction_idx, flag in enumerate(interaction_mask):

            # leave out interactions that are not formed by the current residue
            if not flag:
                continue

            current_t_in2 = np.copy(t_in1)
            current_t_in2[0, len(RESI_TLCS) + interaction_idx] = 1.  # set interaction type
            t_in2.append(current_t_in2)

        t_in2 = np.concatenate(t_in2, axis=0)
        t_in1 = np.repeat(t_in1, len(t_in2), axis=0)  # repeat along batch dim

        prediction = ffnn.predict_on_batch([t_in1, t_in2])
        matrix_line = np.full(len(INTERACTIONS), fill_value=np.nan, dtype=float)
        matrix_line[interaction_mask] = prediction
        out_matrix.append(matrix_line)

    out_matrix = np.array(out_matrix)

    return out_matrix


def plot_results(ffnn: tf.keras.models.Model, merged_dataset: EnvironmentPairList):

    t_in1, t_in2, y_true = merged_dataset.get_training_data()

    plt.rcParams.update({
        "font.size": 7,
        "font.family": "Arial",
        "figure.subplot.left": 0.2,
        "figure.subplot.bottom": 0.15
    })

    y_pred = ffnn.predict_on_batch([t_in1, t_in2])

    corr_s = spearmanr(y_true, y_pred)
    mean_abs_err = np.mean(np.abs(y_pred - y_true))
    mean_squ_err = np.sqrt(np.mean((y_pred - y_true) ** 2))

    print(f"SpR = {corr_s}")

    # Get the residue dependence of error

    fig, ax = plt.subplots()
    fig.suptitle("Prediction error dependence for different residue pairs")
    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)

    error_dict = dict()
    for idx, current_error in enumerate(np.abs(y_pred - y_true)):

        resi1 = merged_dataset.side1_ids[idx][-3:]
        resi2 = merged_dataset.side2_ids[idx][-3:]

        current_key = (resi1, resi2) if resi1 > resi2 else (resi2, resi1)
        error_list = error_dict.get(current_key, list())
        error_list.append(current_error)
        error_dict[current_key] = error_list

    error_dict = {key: np.mean(errors) for key, errors in error_dict.items()}

    error_mx = np.zeros((len(RESI_TLCS), len(RESI_TLCS)))
    for key, current_error in error_dict.items():

        resi1_idx = RESI_TLCS.index(key[0])
        resi2_idx = RESI_TLCS.index(key[1])
        error_mx[resi1_idx, resi2_idx] = error_mx[resi2_idx, resi1_idx] = current_error

        ax.text(resi1_idx, resi2_idx, f"{current_error:.1%}", ha="center", va="center", color="black", fontsize=8)

        if resi1_idx == resi2_idx:  # don't place text in the diagonals 2 times
            continue

        ax.text(resi2_idx, resi1_idx, f"{current_error:.1%}", ha="center", va="center", color="black", fontsize=8)

    ax.imshow(error_mx, cmap="autumn")
    ax.set_xticks(np.arange(len(RESI_TLCS)), labels=RESI_TLCS, rotation=90)
    ax.set_yticks(np.arange(len(RESI_TLCS)), labels=RESI_TLCS)

    plt.savefig(OUTPUT_PATH / "prediction_resi_dependence.svg", dpi=300)
    plt.close(fig)

    # Set 2D histogram

    fig, ax = plt.subplots()
    fig.suptitle("2D histogram of the true and neural network\npredicted LoCoHD scores")
    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)
    hist, x_tick_labels, y_tick_labels = np.histogram2d(
        y_true, y_pred,
        bins=100
    )

    ax.imshow(hist[:, ::-1].T, cmap="hot")
    ax.set_xlabel("true LoCoHD score")
    ax.set_ylabel("predicted LoCoHD score")

    x_tick_labels, y_tick_labels = x_tick_labels[::15], y_tick_labels[::15]  # keep only 100 / 15 ~ 7 ticks
    tick_posi = list(range(0, 100, 15))
    ax.set_xticks(tick_posi, labels=[f"{x:.1%}" for x in x_tick_labels])
    ax.set_yticks(tick_posi, labels=[f"{y:.1%}" for y in y_tick_labels[::-1]])

    legend_labels = list()
    legend_labels.append(f"SpR = {corr_s.statistic:.5f}")
    legend_labels.append(f"MAE = {mean_abs_err:.3%}")
    legend_labels.append(f"RMSE = {mean_squ_err:.3%}")
    legend_handles = [Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0), ] * len(legend_labels)
    ax.legend(
        legend_handles, legend_labels,
        loc="upper right", fancybox=True,
        framealpha=0.7, handlelength=0, handletextpad=0
    )

    plt.savefig(OUTPUT_PATH / "histogram.svg", dpi=300)
    plt.close(fig)

    # Set residue-interaction matrix

    fig, ax = plt.subplots()
    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)
    fig.suptitle("Expected LoCoHD value between interaction-less\nand single interaction environments")

    all_inputs = np.concatenate([t_in1, t_in2], axis=0)
    interaction_dependence_mx = test_interaction_dependence(all_inputs, ffnn)

    ax.imshow(interaction_dependence_mx, cmap="autumn")
    ax.set_xticks(np.arange(len(INTERACTIONS)), labels=INTERACTIONS, rotation=90)
    ax.set_yticks(np.arange(len(RESI_TLCS)), labels=RESI_TLCS)

    for x_idx in range(len(RESI_TLCS)):
        for y_idx in range(len(INTERACTIONS)):

            current_value = interaction_dependence_mx[x_idx, y_idx]
            if np.isnan(current_value):
                continue

            ax.text(y_idx, x_idx, f"{current_value:.1%}", ha="center", va="center", color="black")

    ax.set_aspect(len(INTERACTIONS) / len(RESI_TLCS))

    plt.savefig(OUTPUT_PATH / "resi_interaction_mx.svg", dpi=300)


def main():

    ring_data = get_ring_data()
    merged_dataset = merge_datasets(ring_data)
    t_in1, t_in2, y_true = merged_dataset.get_training_data()

    # np.random.shuffle(y_true)

    ffnn = create_ffnn()
    ffnn.summary()
    ffnn.fit(x=[t_in1, t_in2], y=y_true, batch_size=64, epochs=3, validation_split=0.2)

    plot_results(ffnn, merged_dataset)


if __name__ == "__main__":
    main()
