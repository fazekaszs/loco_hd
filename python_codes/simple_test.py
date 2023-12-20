import numpy as np
import matplotlib.pyplot as plt

from loco_hd import LoCoHD, WeightFunction


def main():

    categories = list(map(str, range(6)))
    deltas = [0., 1., 2., 4., 8., 16., 32., 64.]
    n_of_points = 100

    # lchd = LoCoHD(categories, ("kumaraswamy", [0., 3.0, 2., 5.]))
    weight_function = WeightFunction("hyper_exp", [1., 1.])
    lchd = LoCoHD(categories, weight_function)
    points = np.random.uniform(0., 1., size=(n_of_points, 3))
    sequence1 = np.random.choice(categories, size=n_of_points)

    fig, ax = plt.subplots()
    for idx, delta in enumerate(deltas):

        directions = np.random.normal(0, 1, size=(n_of_points, 3))
        normals = np.sqrt(np.sum(directions ** 2, axis=1, keepdims=True))
        directions = points + delta * directions / normals
        sequence2 = np.random.choice(categories, size=n_of_points)

        lchd_score = lchd.from_coords(sequence1, sequence2, points, directions)
        ax.scatter(idx * np.ones_like(lchd_score), lchd_score, c="red", alpha=0.05)

    ax.set_xticks(list(range(len(deltas))), labels=list(map(str, deltas)))
    ax.set_xlim((-1, len(deltas)))
    ax.set_ylim((-0.05, 1))
    ax.set_xlabel("Displacement Distance")
    ax.set_ylabel("LoCoHD Score")
    plt.show()


if __name__ == "__main__":
    main()
