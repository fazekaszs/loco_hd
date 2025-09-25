from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as tofu

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from pisces_utils import (
    RESI_TLCS, INTERACTIONS, OUTPUT_PATH,
    EnvironmentPairList, get_ring_data, merge_datasets,
    tlc_to_one_hot
)

MM_TO_INCH = 0.0393701


class SiameseNeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.shared1 = nn.Linear(in_features=26, out_features=256)
        self.shared2 = nn.Linear(in_features=256, out_features=128)
        self.ff_out1 = nn.Linear(in_features=128, out_features=64)
        self.ff_out2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, in1, in2):

        vec1 = tofu.gelu(self.shared1(in1))
        vec1 = tofu.gelu(self.shared2(vec1))

        vec2 = tofu.gelu(self.shared1(in2))
        vec2 = tofu.gelu(self.shared2(vec2))

        vec_combined = tofu.gelu(self.ff_out1((vec1 - vec2) ** 2))
        vec_combined = tofu.sigmoid(self.ff_out2(vec_combined))

        return vec_combined[:, 0]


def to_training_data(merged_dataset: EnvironmentPairList) -> Tuple[np.ndarray, ...]:

    t_in1, t_in2 = list(), list()
    for idx in range(len(merged_dataset)):

        side1_tlc_onehot = tlc_to_one_hot(merged_dataset.side1_ids[idx][-3:])
        side2_tlc_onehot = tlc_to_one_hot(merged_dataset.side2_ids[idx][-3:])

        # feature vec = residue one-hot + contacts
        side1_feature_vec = np.concatenate([side1_tlc_onehot, merged_dataset.side1_interactions[idx]])
        side2_feature_vec = np.concatenate([side2_tlc_onehot, merged_dataset.side2_interactions[idx]])

        t_in1.append(side1_feature_vec)
        t_in2.append(side2_feature_vec)

    # Conversion
    t_in1 = np.array(t_in1)
    t_in2 = np.array(t_in2)
    y_true = np.array(merged_dataset.locohd_values)

    # two input tensors, one output tensor
    return t_in1, t_in2, y_true


def train_network(merged_dataset: EnvironmentPairList) -> SiameseNeuralNet:

    t_in1, t_in2, y_true = to_training_data(merged_dataset)
    ff_nn = SiameseNeuralNet()
    optimizer = torch.optim.Adam(ff_nn.parameters(), lr=0.001)

    y_true_mean = np.mean(y_true)
    y_true_var = np.var(y_true, mean=y_true_mean)

    for epoch in range(10):

        n_batches, remainder = divmod(len(t_in1), 128)
        for batch_idx in range(n_batches):

            optimizer.zero_grad()

            idx_start = batch_idx * 128
            idx_end = (batch_idx + 1) * 128
            predictions = ff_nn(
                torch.Tensor(t_in1[idx_start:idx_end]),
                torch.Tensor(t_in2[idx_start:idx_end])
            )

            y_true_tensor = torch.Tensor(y_true[idx_start:idx_end])
            bce = tofu.binary_cross_entropy(predictions, y_true_tensor)
            mse = torch.mean((predictions - torch.Tensor(y_true[idx_start:idx_end])) ** 2)

            bce.backward()
            optimizer.step()
            print(f"\r{epoch}, {batch_idx / n_batches:.3%} ({bce = :.5f}, {mse = :.5f})", end="")
        print()

        with torch.no_grad():

            predictions = ff_nn(
                torch.Tensor(t_in1),
                torch.Tensor(t_in2)
            ).numpy()
            mse = np.mean((predictions - y_true) ** 2)
            r_squared = 1. - mse / y_true_var
            correlation = spearmanr(predictions, y_true).statistic

            print(f"Epoch {epoch}, {mse = :.5f}, {r_squared = :.5f}, {correlation = :.5f}")

    return ff_nn


def test_interaction_dependence(feature_vecs: np.ndarray, ff_nn: SiameseNeuralNet):

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

        with torch.no_grad():
            prediction = ff_nn(torch.Tensor(t_in1), torch.Tensor(t_in2)).numpy()

        matrix_line = np.full(len(INTERACTIONS), fill_value=np.nan, dtype=float)
        matrix_line[interaction_mask] = prediction
        out_matrix.append(matrix_line)

    out_matrix = np.array(out_matrix)

    return out_matrix


def plot_results(ff_nn: SiameseNeuralNet, merged_dataset: EnvironmentPairList):

    t_in1, t_in2, y_true = to_training_data(merged_dataset)

    plt.rcParams.update({
        "font.size": 7,
        "font.family": "Arial",
        "figure.subplot.left": 0.2,
        "figure.subplot.bottom": 0.15
    })

    with torch.no_grad():
        y_pred = ff_nn(torch.Tensor(t_in1), torch.Tensor(t_in2)).numpy()

    # Save the true and predicted values

    df = pd.DataFrame(
        data=np.array([y_true, y_pred]).T,
        columns=["true LoCoHD", "predicted LoCoHD"]
    )
    df.to_csv(OUTPUT_PATH / "true_vs_pred_LoCoHD.tsv", sep="\t")

    # Calculate statistics

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
    interaction_dependence_mx = test_interaction_dependence(all_inputs, ff_nn)

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
    ff_nn = train_network(merged_dataset)
    plot_results(ff_nn, merged_dataset)


if __name__ == "__main__":
    main()
