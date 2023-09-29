import os
import sys
import subprocess as subp
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

RING_FILE_PATH = Path("../ring-3.0.0/ring/bin/ring")
PISCES_DIR_PATH = Path("../../../PycharmProjects/databases/pisces_220222")
OUTPUT_PATH = Path("../workdir/pisces/ring_result")
INTERACTIONS = [
    "HBOND", "VDW", "SSBOND", "IONIC", "PIPISTACK", "PICATION"
]
OUT_FILE_NAME = "collected.pickle"
PISCES_LOCOHD_FILE_PATH = Path("../workdir/pisces/run_2023-02-08-12-50-23/locohd_data.pisces")


def run_ring():
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


def collect_ring_result():
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


def tlc_to_one_hot(tlc: str):
    """
    Converts the three-letter code of an amino acid to a one-hot encoding.

    :param tlc: The three-letter code of the amino acid.
    :return: The one-hot encoded vector.
    """

    all_tlcs = [
        "GLY", "ALA", "VAL", "ILE", "LEU", "PHE",
        "SER", "THR", "TYR", "ASP", "GLU", "ASN",
        "GLN", "TRP", "HIS", "MET", "PRO", "CYS",
        "ARG", "LYS"
    ]

    out = np.zeros(len(all_tlcs))
    out[all_tlcs.index(tlc)] = 1

    return out


def data_for_training(ring_data: Dict[str, np.ndarray]):
    """
    Creates the full dataset for the training of a neural network.

    :param ring_data:
    :return:
    """

    with open(PISCES_LOCOHD_FILE_PATH, "rb") as f:
        locohd_data = pickle.load(f)

    # Merge the LoCoHD dataset (resi1, resi2, lchd score)
    # and the RING dataset (resi keys, contact count vectors):
    merged_data = [
        (*x, ring_data.get(x[0], np.zeros(len(INTERACTIONS))), ring_data.get(x[1], np.zeros(len(INTERACTIONS))))
        for x in locohd_data
    ]

    t_in1, t_in2, t_out = list(), list(), list()  # two input tensors, one output tensor
    for x in merged_data:

        t_in1.append(np.concatenate([tlc_to_one_hot(x[0][-3:]), x[3]]))  # residue1 one-hot + contacts
        t_in2.append(np.concatenate([tlc_to_one_hot(x[1][-3:]), x[4]]))  # residue2 one-hot + contacts
        t_out.append(x[2])  # LoCoHD value

    return np.array(t_in1), np.array(t_in2), np.array(t_out)


def create_ffnn():
    """
    Creates a feedforward neural network model. It should have two inputs and must be symmetric
    to input swapping, since it will try to mimic a metric.

    :return: The neural network model.
    """

    l_in1 = tf.keras.layers.Input(shape=(26, ))
    l_in2 = tf.keras.layers.Input(shape=(26, ))

    l_hidden = tf.keras.layers.Dense(128, activation="relu")
    l_output = tf.keras.layers.Dense(16, activation="sigmoid")
    l_combiner = tf.keras.layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_mean((x[0] - x[1]) ** 2, axis=1) + 1E-10)
    )

    t_vec1 = l_hidden(l_in1)
    t_vec1 = l_output(t_vec1)

    t_vec2 = l_hidden(l_in2)
    t_vec2 = l_output(t_vec2)

    t_combined = l_combiner([t_vec1, t_vec2])

    ffnn = tf.keras.Model(inputs=(l_in1, l_in2), outputs=t_combined)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    ffnn.compile(optimizer=optimizer, loss="mse")

    return ffnn


def main():

    if os.path.exists(OUTPUT_PATH / OUT_FILE_NAME):

        print(f"{str(OUTPUT_PATH / OUT_FILE_NAME)} already exists! Using this file...")
        with open(OUTPUT_PATH / OUT_FILE_NAME, "rb") as f:
            ring_data = pickle.load(f)

    else:

        run_ring()
        ring_data = collect_ring_result()
        with open(OUTPUT_PATH / "collected.pickle", "wb") as f:
            pickle.dump(ring_data, f)

    t_in1, t_in2, t_out = data_for_training(ring_data)

    ffnn = create_ffnn()
    ffnn.summary()
    ffnn.fit(x=[t_in1, t_in2], y=t_out, batch_size=64, epochs=3, validation_split=0.2)

    y_pred = ffnn.predict_on_batch([t_in1, t_in2])

    corr_p, corr_s = pearsonr(t_out, y_pred), spearmanr(t_out, y_pred)

    print(f"Correlation:\nPearson = {corr_p}\nSpearman = {corr_s}")

    fig, ax = plt.subplots()
    ax.scatter(y_pred[::5], t_out[::5], alpha=0.05)

    plt.show()


if __name__ == "__main__":
    main()
